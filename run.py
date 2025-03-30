from ultralytics import YOLO
import cv2

def Train(
        model="yolo11n-seg.pt",
        data="./Floor-plan-segmentation.v17i.yolov11/data.yaml"
):
    model = YOLO(model)

    model.train(
        data=data,
        epochs=50,
        imgsz=640,
        batch=2,
        device="cuda",
        workers=4,
    )

def Atest(filename,img):
    model=YOLO(filename)
    results=model.predict(
        source=img,
        conf=0.5,
        show=True
    )
    for r in results:
        im_array=r.plot()
        cv2.imshow("Prediction",im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mode='test'#'train'
    if mode=='test':
        #导入训练好的模型文件
        model_file="./runs/segment/train/weights/best.pt"
        #导入要测试的图片
        img="./Floor-plan-segmentation.v17i.yolov11/test/images/Engrace-2-B2-6_JPG.rf.70246d2fadb905f98dbfa69696edcfdf.jpg"
        Atest(model_file,img)
    else:
        Train()