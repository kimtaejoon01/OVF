# github_app.py ───────────────────────────────────────────────────────────────
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch, numpy as np, tempfile, os, gdown
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from uuid import uuid4

from preprocess import read_dicom_series, n4_bias_correction, get_bbox, crop
from model import OVFNet
from torchvision.transforms import v2


# ─────────── 상수 ───────────
AGE_MEAN, AGE_STD = 72.78, 9.38
BMD_MEAN, BMD_STD = -3.154, 1.077
MODEL_IDS = {
    False: "172-BN31rBnRj6x9sKZjr7blBZcTrL_rM",   # use_clinical_0.pth
    True : "1TUGx0WfMznyVlejRPZFxWlLZ6O8oOwLG",   # use_clinical_1.pth
}


# ─────────── 유틸 ───────────
def download_model_from_gdrive(file_id, save_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, save_path, quiet=False)

def load_model(use_clinical=False):
    """use_clinical이 True면 임상정보 포함 모델, False면 기본 모델 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = OVFNet(use_clinical=use_clinical).to(device)
    os.makedirs("models", exist_ok=True)
    ck = f"models/use_clinical_{int(use_clinical)}.pth"
    if not os.path.exists(ck):
        with st.spinner("모델 파일 다운로드 중..."):
            download_model_from_gdrive(MODEL_IDS[use_clinical], ck)
    net.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
    net.eval()
    return net, device


@st.cache_data
def load_dicom(files):
    """DICOM 시리즈(.dcm) 파일들을 한 폴더에 저장하고 읽어들여 N4 보정"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        for f in files:
            (p / f.name).write_bytes(f.getvalue())
        img = read_dicom_series(str(p))
        return n4_bias_correction(img)   # (F,H,W)


def slice_uint8(slc):
    sl = np.clip(slc, np.quantile(slc, .05), np.quantile(slc, .95))
    sl = (sl - sl.min()) / (sl.max() - sl.min())
    return (sl * 255).astype(np.uint8)


def draw_points(img_uint8, kps):
    """uint8 Gray → RGB 후, 빨간 점과 번호 표시"""
    img = Image.fromarray(img_uint8).convert("RGB")
    drw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for i, (y,x) in enumerate(kps, start=1):
        r = 4
        drw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
        label = str(i)
        try:
            w, h = drw.textbbox((0,0), label, font=font)[2:]
        except AttributeError:
            w, h = font.getsize(label)
        drw.text((x - w/2, y - h/2), label, fill="yellow", font=font)

    return img


def vis_crop(arr, idx, kp, bb):
    sl = slice_uint8(crop(arr[idx], bb))
    kp_rel = kp - np.array([bb[0], bb[1]])
    return draw_points(sl, kp_rel)


def process_image(arr, idx_list, kp, bb):
    """idx_list 슬라이스 각각을 crop, 전처리 후 (3,C,H,W) 텐서 반환"""
    tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((224,224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                     std =(0.26862954,0.26130258,0.27577711)),
    ])
    outs = []
    for j in idx_list:
        sl = slice_uint8(crop(arr[j], bb))
        outs.append(tfm(np.tile(sl[..., None], (1,1,3))))
    return torch.stack(outs)  # (3, C, H, W)


# ─────────── 메인 ───────────
def main():
    st.title("골다공성 압박골절 예측")

    # ❶ 임상 정보 사용 여부
    use_clinical = st.sidebar.checkbox("임상 정보 사용", value=False)

    # ❷ 임상/비임상 모델을 별도로 캐싱
    @st.cache_resource
    def get_net(flag: bool):
        return load_model(flag)
    net, device = get_net(use_clinical)

    files = st.file_uploader("DICOM 시리즈 업로드(.dcm)", type=["dcm"],
                             accept_multiple_files=True)
    if not files:
        return

    # 파일 시그니처로 업로드 변경 감지
    file_sig = tuple((f.name, f.size) for f in files)
    if "dicom_sig" not in st.session_state or st.session_state.dicom_sig != file_sig:
        st.session_state.arr = load_dicom(files)   # (F,H,W)
        st.session_state.dicom_sig = file_sig
        # 세션 변수 초기화
        st.session_state.sel_frames = []
        st.session_state.kps = []
        st.session_state.canvas_key = str(uuid4())

    arr: np.ndarray = st.session_state.arr
    F, H, W = arr.shape
    st.write(f"업로드한 슬라이스: {F}장")

    # ───── 1단계: 3개 프레임 선택 ─────
    if "sel_frames" not in st.session_state:
        st.session_state.sel_frames = []

    st.subheader("① 분석할 3 프레임 선택")

    # 미리 체크값 읽기
    thumbs = [slice_uint8(arr[i]) for i in range(F)]
    cols = st.columns(5)
    curr_checked = {i: st.session_state.get(f"sel_{i}", False) for i in range(F)}
    selected_count = sum(curr_checked.values())

    # 체크박스 그리기
    for i, img in enumerate(thumbs):
        col = cols[i % 5]
        with col:
            st.image(img, caption=str(i), use_column_width=True)
            disabled_flag = (selected_count >= 3 and not curr_checked[i])
            st.checkbox("선택", key=f"sel_{i}", disabled=disabled_flag)

    # 최종 선택 배열
    selected_idx = [i for i in range(F) if st.session_state.get(f"sel_{i}", False)]
    # 초과 시 앞쪽부터 해제
    if len(selected_idx) > 3:
        for i in selected_idx[3:]:
            st.session_state[f"sel_{i}"] = False
        selected_idx = selected_idx[:3]

    st.session_state.sel_frames = selected_idx
    sel = sorted(st.session_state.sel_frames)

    # 3장 미만/초과 체크
    if len(sel) < 3:
        st.warning("3 장을 선택하거나 업로드해 주세요.")
        st.stop()
    if len(sel) > 3:
        st.warning("현재 3 장까지만 지원합니다. 체크를 해제해 주세요.")
        st.stop()

    # ───── 2단계: 키프레임(중간) ─────
    kf = sel[1]
    st.markdown(f"**선택된 프레임:** {sel} → 키프레임 = {kf}")

    # ───── 3단계: 키포인트 입력 ─────
    if "kps" not in st.session_state:
        st.session_state.kps = []

    method = st.radio("키포인트 입력 방법", ["이미지 클릭", "좌표 직접 입력"])

    if method == "이미지 클릭":
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = str(uuid4())
        if len(st.session_state.kps) < 6:
            bg = Image.fromarray(slice_uint8(arr[kf])).convert("RGB")
            res = st_canvas(background_image=bg, height=H, width=W,
                            drawing_mode="point", point_display_radius=4,
                            stroke_width=3, fill_color="rgba(255,0,0,0.8)",
                            key=st.session_state.canvas_key)
            if res.json_data:
                objs = res.json_data["objects"][:6]
                st.session_state.kps = [
                    (int(round(o["top"])), int(round(o["left"]))) for o in objs
                ]
                if len(st.session_state.kps) >= 6:
                    st.rerun()
        if len(st.session_state.kps) == 6:
            st.image(
                draw_points(slice_uint8(arr[kf]), st.session_state.kps),
                caption="키포인트 완료"
            )
        st.write(f"선택된 키포인트: {len(st.session_state.kps)} / 6")

    else:  # 좌표 직접 입력
        c1, c2 = st.columns(2)
        with c1:
            xs = [st.number_input(f"X{i+1}", step=1, key=f"xi{i}") for i in range(6)]
        with c2:
            ys = [st.number_input(f"Y{i+1}", step=1, key=f"yi{i}") for i in range(6)]
        if st.button("좌표 적용"):
            st.session_state.kps = [(int(ys[i]), int(xs[i])) for i in range(6)]
            st.rerun()

    # 키포인트 초기화
    if st.button("키포인트 초기화"):
        st.session_state.kps = []
        st.session_state.canvas_key = str(uuid4())
        st.rerun()

    # ───── 4단계: 예측 ─────
    if len(st.session_state.kps) == 6:

        # 임상 정보 입력 (use_clinical==True이면)
        meta = None
        if use_clinical:
            st.markdown("### 임상 정보 입력")
            # 당장 입력-버튼 간 rerun 필요하면 session_state로 키 지정
            c_age = st.number_input("나이",  0, 150, 65, key="c_age")
            c_gen = st.radio      ("성별",  ["남성","여성"], key="c_gender")
            c_bmd = st.number_input("BMD", -6.0, 2.0, -2.5, key="c_bmd")
            c_pre = st.checkbox   ("이전 약물 복용", key="c_pre")
            c_post= st.checkbox   ("현재 약물 복용", key="c_post")

        if st.button("예측하기"):
            kp_np = np.array(st.session_state.kps, np.int32)
            bb = get_bbox(kp_np, 1.5)

            # 실제 meta 구성
            if use_clinical:
                meta = torch.tensor([
                    (c_age - AGE_MEAN) / AGE_STD,
                    1 if c_gen == "여성" else 0,
                    (c_bmd - BMD_MEAN) / BMD_STD,
                    float(c_pre),
                    float(c_post)
                ], dtype=torch.float32).to(device)

            x = process_image(arr, sel, kp_np, bb).to(device)
            with torch.no_grad():
                # 3개 슬라이스 각각 추론 → 확률 평균
                probs = [
                    torch.sigmoid(
                        net(x[i, None], meta[None] if meta is not None else None
                            ).squeeze()
                    ) for i in range(3)
                ]
                prob = torch.stack(probs).mean().item()

            st.subheader("예측 결과")
            st.write(f"골다공성 압박골절 확률: **{prob:.2%}**")
            (st.error if prob > 0.5 else st.success)(
                "위험이 높습니다." if prob > 0.5 else "위험이 낮습니다."
            )


if __name__ == "__main__":
    main()
