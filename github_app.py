# app.py ───────────────────────────────────────────────────────────────
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch, numpy as np, tempfile, requests, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from uuid import uuid4
import gdown 

from preprocess import read_dicom_series, n4_bias_correction, get_bbox, crop
from model import OVFNet
from torchvision.transforms import v2


# ─────────── 상수 ───────────
AGE_MEAN, AGE_STD = 72.77551020408163, 9.378789520714859
BMD_MEAN, BMD_STD = -3.154285714285714, 1.0773905938902932

# Google Drive 파일 ID  (공유 링크에서 추출)
MODEL_IDS = {
    False: "172-BN31rBnRj6x9sKZjr7blBZcTrL_rM",   # use_clinical_0.pth
    True : "1TUGx0WfMznyVlejRPZFxWlLZ6O8oOwLG",   # use_clinical_1.pth
}


# ─────────── 유틸 ───────────
def download_model_from_gdrive(file_id: str, save_path: str):
    """gdown으로 대용량 Google Drive 파일 다운로드"""
    url = f"https://drive.google.com/uc?id={file_id}"
    # gdown이 알아서 confirm 토큰 처리
    gdown.download(url, save_path, quiet=False)


def load_model(use_clinical: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OVFNet(use_clinical=use_clinical).to(device)

    os.makedirs("models", exist_ok=True)
    ck_path = f"models/use_clinical_{int(use_clinical)}.pth"

    if not os.path.exists(ck_path):
        with st.spinner("모델 파일을 다운로드 하는 중입니다..."):
            download_model_from_gdrive(MODEL_IDS[use_clinical], ck_path)

    # ─ 변경된 부분 ─
    state_dict = torch.load(ck_path, map_location=device, weights_only=False)
    net.load_state_dict(state_dict)
    # ────────────────

    net.eval()
    return net, device


@st.cache_data
def load_dicom(files):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        for f in files:
            (p / f.name).write_bytes(f.getvalue())
        img = read_dicom_series(str(p))
        return n4_bias_correction(img)  # (F,H,W) ndarray


def slice_uint8(slc):
    sl = np.clip(slc, np.quantile(slc, 0.05), np.quantile(slc, 0.95))
    sl = (sl - sl.min()) / (sl.max() - sl.min())
    return (sl * 255).astype(np.uint8)


def draw_points(img_uint8, kps):
    """uint8 Gray → RGB + 빨간 점 + 노란 번호 (중앙 정렬, Pillow 7~10 호환)"""
    img = Image.fromarray(img_uint8).convert("RGB")
    drw = ImageDraw.Draw(img)

    try:  # 시스템 기본 글꼴로
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for i, (yy, xx) in enumerate(kps, start=1):
        r = 4
        drw.ellipse((xx - r, yy - r, xx + r, yy + r), outline="red", width=2)

        label = str(i)
        # ─ 글자 폭·높이 구하기 (버전별 fallback) ─
        try:
            # Pillow ≥10 : textbbox
            bbox = drw.textbbox((0, 0), label, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Pillow <10 : font.getsize
            w, h = font.getsize(label)

        drw.text((xx - w / 2, yy - h / 2), label, fill="yellow", font=font)

    return img


def vis_crop(arr, kf, kp, bb):
    sl = slice_uint8(crop(arr[kf], bb))
    kp_rel = kp - np.array([bb[0], bb[1]])
    return draw_points(sl, kp_rel)


def process_image(arr, kf, kp, bb):
    tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    outs = []
    for j in [kf - 1, kf, kf + 1]:
        sl = slice_uint8(crop(arr[j], bb))
        outs.append(tfm(np.tile(sl[..., None], (1, 1, 3))))
    return torch.stack(outs)


# ─────────── 메인 ───────────
def main():
    st.title("골다공성 압박골절 예측")

    use_clinical = st.sidebar.checkbox("임상 정보 사용", value=False)

    @st.cache_resource
    def get_net():
        return load_model(use_clinical)
    net, device = get_net()

    files = st.file_uploader("DICOM 시리즈 업로드(.dcm)", type=["dcm"],
                             accept_multiple_files=True)
    if not files:
        return

    arr = load_dicom(files)                 # (F,H,W)
    F, H, W = arr.shape
    st.write(f"Frames={F}, Height={H}, Width={W}")

    # ─ 키프레임 ─
    if "kf" not in st.session_state:
        st.session_state.kf = F // 2

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("◀"):
            st.session_state.kf = max(1, st.session_state.kf - 1)
    with col2:
        st.write(f"{st.session_state.kf} / {F - 1}")
    with col3:
        if st.button("▶"):
            st.session_state.kf = min(F - 2, st.session_state.kf + 1)
    kf = st.session_state.kf

    # ─ 키포인트 저장소 ─
    if "kps" not in st.session_state:
        st.session_state.kps = []

    # ─ 입력 방법 선택 ─
    method = st.radio("키포인트 입력 방법", ["이미지 클릭", "좌표 직접 입력"])

    # ───────── 이미지 클릭 모드 ─────────
    if method == "이미지 클릭":
        # 6개 미만이면 캔버스 활성
        if len(st.session_state.kps) < 6:
            if "canvas_key" not in st.session_state:
                st.session_state.canvas_key = str(uuid4())

            bg_img = draw_points(slice_uint8(arr[kf]), st.session_state.kps)
            canvas_res = st_canvas(
                background_image=bg_img,
                height=H, width=W,
                drawing_mode="point",
                point_display_radius=4,
                stroke_width=3,
                fill_color="rgba(255,0,0,0.8)",
                key=st.session_state.canvas_key,
            )

            # 새 점 저장
            if canvas_res.json_data:
                objs = canvas_res.json_data["objects"]
                if len(objs) > len(st.session_state.kps):
                    y, x = int(round(objs[-1]["top"])), int(round(objs[-1]["left"]))
                    if len(st.session_state.kps) < 6:
                        st.session_state.kps.append((y, x))

        # 6개 완료 → 정적 이미지 (점 포함) 표시
        if len(st.session_state.kps) == 6:
            st.image(draw_points(slice_uint8(arr[kf]), st.session_state.kps),
                     caption="선택된 키포인트 완료")

        st.write(f"선택된 키포인트: {len(st.session_state.kps)} / 6")

    # ───────── 좌표 직접 입력 모드 ─────────
    else:
        cols_x, cols_y = st.columns(2)
        with cols_x:
            xs = [st.number_input(f"X{i+1}", step=1, key=f"xi{i}") for i in range(6)]
        with cols_y:
            ys = [st.number_input(f"Y{i+1}", step=1, key=f"yi{i}") for i in range(6)]

        if st.button("좌표 적용"):
            st.session_state.kps = [(int(ys[i]), int(xs[i])) for i in range(6)]
            st.rerun()

    # ── 초기화 ──
    if st.button("키포인트 초기화"):
        st.session_state.kps = []
        st.session_state.canvas_key = str(uuid4())
        st.rerun()

    # ── 6 점 완료 후 동작 ──
    if len(st.session_state.kps) == 6:
        kp_np = np.array(st.session_state.kps, np.int32)
        bb = get_bbox(kp_np, 1.5)

        if st.button("크롭된 이미지 보기"):
            st.image(vis_crop(arr, kf, kp_np, bb), caption="크롭 확인")

        if st.button("예측하기"):
            meta = None
            if use_clinical:
                age = st.number_input("나이", 0, 150, 65)
                gender = st.radio("성별", ["남성", "여성"])
                bmd = st.number_input("BMD", value=-2.5)
                pre_drug = st.checkbox("이전 약물 복용")
                post_drug = st.checkbox("현재 약물 복용")
                meta = torch.tensor([
                    (age - AGE_MEAN) / AGE_STD,
                    1 if gender == "여성" else 0,
                    (bmd - BMD_MEAN) / BMD_STD,
                    float(pre_drug), float(post_drug)], dtype=torch.float32).to(device)

            x = process_image(arr, kf, kp_np, bb).to(device)
            with torch.no_grad():
                probs = [torch.sigmoid(
                    net(x[i, None], meta[None] if use_clinical else None
                        ).squeeze()) for i in range(3)]
                prob = torch.stack(probs).mean().item()

            st.subheader("예측 결과")
            st.write(f"골다공성 압박골절 확률: **{prob:.2%}**")
            (st.error if prob > 0.5 else st.success)(
                "위험이 높습니다." if prob > 0.5 else "위험이 낮습니다."
            )


if __name__ == "__main__":
    main()
