import numpy as np
from fundrive.download import simple_download
from fundrive.drives.oss import public_oss_url
from funmodel.core.predict import ImagePredictModel
from funmodel.dwpose.core import util
from funmodel.dwpose.core.wholebody import Wholebody


class DWposePredict(ImagePredictModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_estimation = None
        self.load()

    def load(self, *args, **kwargs):
        onnx_det = f"{self.cache_path}/yolox_l.onnx"
        onnx_pose = f"{self.cache_path}/dw-ll_ucoco_384.onnx"
        simple_download(
            url=public_oss_url(path="models/dwpose/yolox_l.onnx"), filepath=onnx_det
        )
        simple_download(
            url=public_oss_url(path="models/dwpose/dw-ll_ucoco_384.onnx"),
            filepath=onnx_pose,
        )
        self.pose_estimation = Wholebody(onnx_det=onnx_det, onnx_pose=onnx_pose)

    def draw_image(self, image, result, *args, **kwargs):
        canvas = np.zeros(shape=(result["height"], result["width"], 3), dtype=np.uint8)
        canvas = util.draw_bodypose(canvas, result["bodies"])
        canvas = util.draw_handpose(canvas, result["hand1"])
        canvas = util.draw_handpose(canvas, result["hand2"])
        canvas = util.draw_facepose(canvas, result["faces"])
        return canvas

    def predict(self, ori_img, draw=False, *args, **kwargs):
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        candidate, subset = self.pose_estimation(ori_img)
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        pose = dict(
            height=H,
            width=W,
            bodies=candidate[:, :18],
            hand1=candidate[:, 92:113],
            hand2=candidate[:, 113:134],
            faces=candidate[:, 24:92],
            foots=candidate[:, 18:24],
            nums=candidate.shape[0],
        )

        if draw:
            ori_img = self.draw_image(image=ori_img, result=pose)
        return pose, ori_img
