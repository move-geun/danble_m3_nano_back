# -*- coding: utf-8 -*-
"""
FastAPI 서버 - Flatlay 이미지 생성 API
"""

import os
import json
import tempfile
import shutil
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import requests
import numpy as np

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Flatlay Image Generator API", version="1.0.0")

# CORS 설정 (프론트엔드 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 파일 저장 디렉토리
TEMP_DIR = Path(tempfile.gettempdir()) / "m3_nano_api"
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = TEMP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 템플릿 이미지 경로 (고정)
# 배포 시 case.png 파일을 backend/ 디렉토리에 배치하거나 환경 변수로 경로 지정
# 스크립트 위치 기준으로 상대 경로 계산
SCRIPT_DIR = Path(__file__).parent
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", str(SCRIPT_DIR / "case.png"))

# 고정 Rulebook (tmp_input/rulebook_nano.txt 내용)
RULEBOOK_TEXT = """Placement Criteria
- [**No shadow, flat single solid background**]
- [** Never place things on the edge.**]
- [**offWhite solid using background color, (240, 240, 236) #F0F0EC**]
- The entire product should be placed within the canvas.
(Only if the product is provided, the following items apply.)
- Coat: Spread flat in the center of the jacket (natural sleeves and hem)
- Underwear: Halfway down the right side and place it on the jacket
- Shoes: Two pairs in one direction on the bottom left corner (maximum air shot)
- Socks: Only if the product is provided, Two pairs on the right side of the shoe (air shot)
- Put your jacket in your jacket and flip it over
- Place entire items in a way that does not cut neatly within margin criteria
- Use only the items in the picture
- There's no light at all
- Clothes wrinkles, shades, textures, and colors are the same as real clothes
- Keep your jacket flat
- Place the outer layer naturally so that only one side is open
- The sleeves and collar are also organized using the actual texture of the fabric
- The whole structure is as orderly as it used to be
- Add adequate padding/margin around the edge
- Minimize top, bottom, left and right margins to ensure clear visibility of the product"""


# 요청 모델
class ProductItem(BaseModel):
    type: Optional[str] = None
    sub_type: Optional[str] = ""
    image_url: str
    
    class Config:
        extra = "allow"  # 추가 필드 허용


class GenerateRequest(BaseModel):
    style_id: Optional[str] = None
    products: List[Dict[str, Any]]  # 제품 배열 (유연한 형태 지원)


# 유틸리티 함수들 (main.py에서 가져옴)
def _translate_sub_type_to_english(sub_type: str) -> str:
    translation_map = {
        "벨트": "belt", "belt": "belt",
        "모자": "hat", "캡": "hat", "hat": "hat",
        "스카프": "scarf", "scarf": "scarf",
        "넥타이": "tie", "tie": "tie",
        "목도리": "scarf",
        "가방": "bag", "bag": "bag", "백": "bag",
        "양말": "socks", "socks": "socks",
    }
    if not sub_type:
        return sub_type or ""
    sub_lower = sub_type.lower()
    if sub_lower in translation_map:
        return translation_map[sub_lower]
    for korean, english in translation_map.items():
        if korean in sub_lower or sub_lower in korean:
            return english
    return sub_type


def _url_to_pil_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 이미지 다운로드 실패: {url[:50]}... - {type(e).__name__}: {e}")
        return None


def _pil_image_to_blob(img: Image.Image) -> types.Blob:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return types.Blob(mime_type="image/png", data=buffer.getvalue())


def _replace_background_color(img: Image.Image, target_color: tuple = (240, 241, 236), threshold: int = 5) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    img_array = np.array(img)
    
    # 좌측 상단 픽셀 색상 샘플링
    sample_size = 3
    top_left_region = img_array[:sample_size, :sample_size, :3]
    bg_color = top_left_region.mean(axis=(0, 1))
    
    # 색상 차이 계산 (유클리드 거리)
    color_diff = np.sqrt(
        np.sum((img_array[:, :, :3] - bg_color) ** 2, axis=2)
    )
    
    # 배경색과 유사한 픽셀 마스크
    bg_mask = color_diff < threshold
    
    # 배경 픽셀을 타겟 색상으로 교체
    result_array = img_array.copy()
    result_array[bg_mask, 0] = target_color[0]
    result_array[bg_mask, 1] = target_color[1]
    result_array[bg_mask, 2] = target_color[2]
    result_array[bg_mask, 3] = 255
    
    return Image.fromarray(result_array, "RGBA")


def _composite_on_template(generated_img: Image.Image, template_path: str, output_path: str, target_size: tuple = (648, 681)) -> str:
    template = Image.open(template_path).convert("RGBA")
    
    if generated_img.mode != "RGBA":
        generated_img = generated_img.convert("RGBA")
    
    # 먼저 리사이즈 (템플릿에 맞게)
    generated_img = generated_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 배경색 교체
    background_color = (240, 241, 236)
    generated_img = _replace_background_color(generated_img, target_color=background_color, threshold=5)
    
    # 중앙 정렬
    paste_x = (template.width - generated_img.width) // 2
    paste_y = (template.height - generated_img.height) // 2 + 10
    
    # 합성
    result = template.copy()
    result.paste(generated_img, (paste_x, paste_y), generated_img)
    
    result.save(output_path, "PNG")
    return output_path


def _save_image_with_background(image: Image.Image, out_path: str, target_size: tuple, background_color: tuple) -> str:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # 리사이즈
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 배경색 교체
    image = _replace_background_color(image, target_color=background_color, threshold=5)
    
    # 저장
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path, "PNG")
    return out_path


def _build_prompt(product_images: List[Dict], top_products: List, bottom_products: List, 
                  shoe_products: List, etc_products: List, system_text: Optional[str] = None) -> str:
    """프롬프트 생성"""
    top_images_list = "\n".join([f"- {item[1]}" for item in top_products])
    bottom_images_list = "\n".join([f"- {item[1]}" for item in bottom_products])
    shoe_images_list = "\n".join([f"- {item[1]}" for item in shoe_products])
    etc_images_list = "\n".join([f"- {item[1]}" for item in etc_products])
    
    user_text_template = """Create a flatlay image of the following outfit items.

TOPS:
{{top_images_list}}

BOTTOMS:
Here are the pants:
{{bottom_images_list}}

ACCESSORIES:
Shoes:
{{shoe_images_list}}

Other accessories (bag / belt / hat / scarf / socks / etc):
{{etc_images_list}}

Follow the system rulebook strictly.
Create the layered outfit, place pants overlapping the viewer-right sleeve, fold pants as defined, and place accessories in their correct regions.
Preserve all product textures, colors, silhouettes, and proportions exactly.
"""
    
    user_text = user_text_template.replace("{{top_images_list}}", top_images_list)
    user_text = user_text.replace("{{bottom_images_list}}", bottom_images_list)
    user_text = user_text.replace("{{shoe_images_list}}", shoe_images_list)
    user_text = user_text.replace("{{etc_images_list}}", etc_images_list)
    
    if system_text:
        user_text = system_text + "\n\n" + user_text
    
    return user_text


@app.get("/")
async def root():
    return {"message": "Flatlay Image Generator API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/generate")
async def generate_image(request: GenerateRequest):
    """이미지 생성 API"""
    try:
        # API 키 확인
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        client = genai.Client(api_key=api_key)
        
        # 제품 이미지 수집
        product_images: List[Dict] = []
        top_products = []
        bottom_products = []
        shoe_products = []
        etc_products = []
        
        for product in request.products:
            # 딕셔너리 형태로 처리
            if not isinstance(product, dict):
                continue
                
            image_url = product.get("image_url")
            if not image_url:
                continue
            
            pil_image = _url_to_pil_image(image_url)
            if pil_image is None:
                continue
            
            product_type = product.get("type", "").lower()
            sub_type = product.get("sub_type", "") or ""
            
            item_type_map = {
                "top": "top",
                "bottom": "pants",
                "outer": "outer",
                "shoes": "shoes",
                "bag": "bag",
                "etc": "etc"
            }
            
            item_type = item_type_map.get(product_type, "etc")
            
            if item_type == "top":
                top_products.append((item_type, sub_type, product_type))
            elif item_type == "pants":
                bottom_products.append((item_type, sub_type, product_type))
            elif item_type == "shoes":
                shoe_products.append((item_type, sub_type, product_type))
            else:
                etc_products.append((item_type, _translate_sub_type_to_english(sub_type), product_type))
            
            product_images.append({
                "image": pil_image,
                "type": item_type,
                "sub_type": sub_type
            })
        
        if not product_images:
            raise HTTPException(status_code=400, detail="No valid product images found")
        
        # 프롬프트 생성 (고정 rulebook 사용)
        user_text = _build_prompt(
            product_images, top_products, bottom_products, 
            shoe_products, etc_products, RULEBOOK_TEXT
        )
        
        # API 호출
        api_parts = [types.Part(text=user_text)]
        for product_info in product_images:
            blob = _pil_image_to_blob(product_info["image"])
            api_parts.append(types.Part(inline_data=blob))
        
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=api_parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1")
            )
        )
        
        # 이미지 추출
        if not response.candidates or not response.candidates[0].content.parts:
            raise HTTPException(status_code=500, detail="No image generated")
        
        image_part = response.candidates[0].content.parts[0]
        if not hasattr(image_part, 'inline_data') or not image_part.inline_data:
            raise HTTPException(status_code=500, detail="Invalid image response")
        
        # 이미지 데이터 추출
        image_data = image_part.inline_data.data
        generated_img = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 이미지 처리 (항상 템플릿 사용)
        style_id = request.style_id or "unknown"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_size = (648, 681)
        background_color = (240, 241, 236)
        
        # 고정된 case.png 템플릿 사용
        template_path = Path(TEMPLATE_PATH)
        if template_path.exists():
            output_path = OUTPUT_DIR / f"{ts}_{style_id}_final.png"
            _composite_on_template(generated_img, str(template_path), str(output_path), target_size)
        else:
            # 템플릿 파일이 없으면 일반 처리 (경고 로그)
            import logging
            logging.warning(f"템플릿 파일을 찾을 수 없습니다: {template_path} (절대 경로: {template_path.resolve()})")
            output_path = OUTPUT_DIR / f"{ts}_{style_id}.png"
            _save_image_with_background(generated_img, str(output_path), target_size, background_color)
        
        # 토큰 사용량 계산
        usage_info = None
        if hasattr(response, 'usage_metadata'):
            usage_info = response.usage_metadata
        elif hasattr(response, 'usage'):
            usage_info = response.usage
        
        token_info = {}
        if usage_info:
            prompt_tokens = getattr(usage_info, 'prompt_token_count', None) or getattr(usage_info, 'prompt_tokens', None) or 0
            candidates_tokens = getattr(usage_info, 'candidates_token_count', None) or getattr(usage_info, 'candidates_tokens', None) or 0
            total_tokens = getattr(usage_info, 'total_token_count', None) or getattr(usage_info, 'total_tokens', None) or (prompt_tokens + candidates_tokens)
            
            input_cost_usd = (prompt_tokens / 1_000_000) * 2.00 + len(product_images) * 0.0011
            output_cost_usd = (candidates_tokens / 1_000_000) * 120.00
            total_cost_usd = input_cost_usd + output_cost_usd
            
            token_info = {
                "prompt_tokens": prompt_tokens,
                "candidates_tokens": candidates_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(total_cost_usd, 6)
            }
        
        return JSONResponse(content={
            "success": True,
            "image_url": f"/api/download/{output_path.name}",
            "filename": output_path.name,
            "tokens": token_info,
            "style_id": style_id
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/download/{filename}")
async def download_image(filename: str):
    """생성된 이미지 다운로드"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="image/png"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

