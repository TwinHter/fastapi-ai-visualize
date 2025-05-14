import io
import matplotlib
matplotlib.use('Agg')  # Dùng backend không GUI
import matplotlib.pyplot as plt
from fastapi import FastAPI, Response, BackgroundTasks, APIRouter
from fastapi.responses import StreamingResponse
import images.eda as eda
import images.compare as compare

router = APIRouter()
@router.get("/eda/area_item", response_class=StreamingResponse)
def get_area_item(area: str, item: str, background_tasks: BackgroundTasks):
    img_buf = eda.eda_area_item(area, item)
    img_buf.seek(0)
    background_tasks.add_task(img_buf.close)  # đảm bảo giải phóng bộ nhớ sau khi gửi xong
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return StreamingResponse(img_buf, media_type='image/png', headers=headers)

@router.get("/eda/area", response_class=StreamingResponse)
def get_area(area: str, background_tasks: BackgroundTasks):
    img_buf = eda.eda_area(area)
    img_buf.seek(0)
    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return StreamingResponse(img_buf, media_type='image/png', headers=headers)

@router.get("/compare", response_class=StreamingResponse)
def get_compare(metric:str, type:str, background_tasks: BackgroundTasks):
    img_buf1 = compare.compare_models_metrics_separate(metric, type)
    img_buf1.seek(0)
    background_tasks.add_task(img_buf1.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return StreamingResponse(img_buf1, media_type='image/png', headers=headers)
