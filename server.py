from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import pandas as pd
import uuid
import os
import base64
from io import BytesIO

app = FastAPI()

# ---- CORS 설정 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 저장폴더 생성
os.makedirs("analyzed_images", exist_ok=True)


# ---- CSV → 그래프 생성 ----
def analyze_csv(file_bytes, filename):
    df = pd.read_csv(BytesIO(file_bytes))

    plt.figure(figsize=(8, 4))
    df_count = df.count()
    df_count.plot(kind="bar")
    plt.title(f"{filename} 컬럼별 데이터 개수")
    plt.tight_layout()

    img_path = f"analyzed_images/{uuid.uuid4()}.png"
    plt.savefig(img_path)
    plt.close()

    return img_path


# ---- 이미지 파일 그대로 Base64로 반환 ----
def encode_image_raw(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")


# ---- 업로드/분석 API ----
@app.post("/analyze")
async def analyze(
    domestic_trend: list[UploadFile] = File(default=[]),
    international_trend: list[UploadFile] = File(default=[]),
    domestic_review: list[UploadFile] = File(default=[]),
    international_review: list[UploadFile] = File(default=[])
):
    all_files = {
        "domestic_trend": domestic_trend,
        "international_trend": international_trend,
        "domestic_review": domestic_review,
        "international_review": international_review
    }

    images_response = []

    # ---- 모든 파일을 순회하며 분석 ----
    for category, files in all_files.items():
        for file in files:
            file_bytes = await file.read()

            # CSV → 분석 후 그래프 이미지 생성
            if file.filename.endswith(".csv"):
                img_path = analyze_csv(file_bytes, file.filename)

                with open(img_path, "rb") as f:
                    base64_img = base64.b64encode(f.read()).decode("utf-8")

                images_response.append({
                    "title": f"{category} : {file.filename}",
                    "data": f"data:image/png;base64,{base64_img}"
                })

            # 이미지 파일 → 그대로 반환
            elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                base64_img = encode_image_raw(file_bytes)

                images_response.append({
                    "title": f"{category} : {file.filename}",
                    "data": f"data:image/png;base64,{base64_img}"
                })

            else:
                return JSONResponse({"error": f"지원하지 않는 파일 형식: {file.filename}"}, status_code=400)

    # ---- 간단한 summary 생성 ----
    summary_text = f"총 {len(images_response)}개의 자료를 분석했습니다."

    return {
        "images": images_response,
        "summary": summary_text
    }
