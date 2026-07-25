import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import shutil
import os

app = FastAPI()

# تفعيل CORS لمنع مشكلة Failed to fetch عند اتصال المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تهيئة النموذج (يعمل تلقائياً على كارت الشاشة GPU إذا كان متاحاً)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("جاري تحميل نموذج الذكاء الاصطناعي لـ Vedtoria...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=dtype, 
    variant="fp16" if device == "cuda" else None
)
pipe.to(device)
pipe.enable_model_cpu_offload()

def enhance_and_upscale_frames(frames):
    """
    معالجة الفريمات لتقليل التشويش، رفع الحدة، وتجنب البهتان.
    """
    enhanced_frames = []
    for frame in frames:
        img = np.array(frame)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # إزالة التشويش
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
        
        # تحسين الحدة والوضوح
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # تكبير الأبعاد لتقليل البكسلة
        height, width, _ = sharpened.shape
        upscaled = cv2.resize(sharpened, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        final_frame = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        enhanced_frames.append(final_frame)
        
    return enhanced_frames

@app.post("/generate-video/")
async def generate_video(file: UploadFile = File(...)):
    temp_image_path = f"temp_{file.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    image = load_image(temp_image_path)
    image = image.resize((1024, 576))
    
    generator = torch.manual_seed(42)
    output = pipe(image, decode_chunk_size=8, generator=generator)
    raw_frames = output.frames[0]
    
    processed_frames = enhance_and_upscale_frames(raw_frames)
    
    output_video_path = "output_vedtoria.mp4"
    export_to_video(processed_frames, output_video_path, fps=8)
    
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    return FileResponse(output_video_path, media_type="video/mp4", filename="vedtoria_video.mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
