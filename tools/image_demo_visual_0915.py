'''
if you use two-stage detector, such as faster rcnn,please change the codes :
1. mmdet/models/detectors/two_stage.py

    def extract_feat(self, img):
    """Directly extract features from the backbone+neck
    """
    x_backbone = self.backbone(img)
    if self.with_neck:
        x_fpn = self.neck(x_backbone)
    return x_backbone,x_fpn

and:

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x_backbone,x_fpn = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x_fpn, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x_fpn, proposal_list, img_metas, rescale=rescale),x_backbone,x_fpn

2.mmdet/apis/inference.py

    def inference_detector(model, img):
    .......
            # forward the model
        with torch.no_grad():
            result,x_backbone,x_fpn= model(return_loss=False, rescale=True, **data)
        return result,x_backbone,x_fpn

if you use other detectors, it is easy to achieve it like this

'''



from mmdet.apis import inference_detector, init_detector
import cv2
import numpy as np
import time
import torch
import os

def main():

    config = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = '/home/maer/Workspace/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth'
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    
    # test images
    images_dir = 'test_images'
    for i in os.listdir(images_dir):
        img_path = os.path.join(images_dir, i)
        img_name = i.split('.')[0]
        image = cv2.imread(img_path)
        height, width, channels = image.shape
        result, x_backbone, x_laterals, x_sum_laterals, x_fpn = inference_detector(model, img_path)

        fm_save_path = os.path.join('feature_map', img_name)
        if not os.path.exists(fm_save_path):
            os.makedirs(fm_save_path)
	
        feature_index = 0
        for feature in x_backbone:
            feature_index += 1
            Stage_save_path = os.path.join(fm_save_path, 'Stage_'+str(feature_index))
            if not os.path.exists(Stage_save_path):
                os.makedirs(Stage_save_path)
            P = torch.sigmoid(feature)
            #P = feature
            P = P.cpu().detach().numpy()
            P = np.maximum(P, 0)
            P = (P - np.min(P)) / (np.max(P) - np.min(P))
            P = P.squeeze(0)
        
            c_list = [81,170,228,141]
            for c in c_list:
                P_ = P[c, ...]
                cam = cv2.resize(P_, (width, height))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap / np.max(heatmap)
                heatmap_image = np.uint8(255 * heatmap)

                cv2.imwrite(Stage_save_path + '/'+ str(c)+'_heatmap.jpg', heatmap_image)
                result = cv2.addWeighted(image, 1, heatmap_image, 0.5, 0)
                cv2.imwrite(Stage_save_path + '/'+ str(c) + '_result.jpg', result)


        feature_index = 1
        for feature in x_laterals:
            feature_index += 1
            M_save_path = os.path.join(fm_save_path, 'M_'+str(feature_index))
            if not os.path.exists(M_save_path):
                os.makedirs(M_save_path)
            P = torch.sigmoid(feature)
            #P = feature
            P = P.cpu().detach().numpy()
            P = np.maximum(P, 0)
            P = (P - np.min(P)) / (np.max(P) - np.min(P))
            P = P.squeeze(0)
            #print(P.shape)

            C = P.shape[0]
            for c in range(C):
                P_ = P[c, ...] 
            

                cam = cv2.resize(P_, (width, height))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap / np.max(heatmap)
                heatmap_image = np.uint8(255 * heatmap)

                cv2.imwrite(M_save_path + '/'+ str(c)+'_heatmap.jpg', heatmap_image)
                result = cv2.addWeighted(image, 0.8, heatmap_image, 0.5, 0)
                cv2.imwrite(M_save_path + '/'+ str(c) + '_result.jpg', result)

        feature_index = 1
        for feature in x_sum_laterals:
            feature_index += 1
            S_save_path = os.path.join(fm_save_path, 'S_'+str(feature_index))
            if not os.path.exists(S_save_path):
                os.makedirs(S_save_path)
            P = torch.sigmoid(feature)
            #P = feature
            P = P.cpu().detach().numpy()
            P = np.maximum(P, 0)
            P = (P - np.min(P)) / (np.max(P) - np.min(P))
            P = P.squeeze(0)
            #print(P.shape)

            C = P.shape[0]
            for c in range(C):
                P_ = P[c, ...] 
            

                cam = cv2.resize(P_, (width, height))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap / np.max(heatmap)
                heatmap_image = np.uint8(255 * heatmap)

                cv2.imwrite(S_save_path + '/'+ str(c)+'_heatmap.jpg', heatmap_image)
                result = cv2.addWeighted(image, 0.8, heatmap_image, 0.5, 0)
                cv2.imwrite(S_save_path + '/'+ str(c) + '_result.jpg', result)

        feature_index = 1
        for feature in x_fpn:
            feature_index += 1
            P_save_path = os.path.join(fm_save_path, 'P_'+str(feature_index))
            if not os.path.exists(P_save_path):
                os.makedirs(P_save_path)
            P = torch.sigmoid(feature)
            #P = feature
            P = P.cpu().detach().numpy()
            P = np.maximum(P, 0)
            P = (P - np.min(P)) / (np.max(P) - np.min(P))
            P = P.squeeze(0)
            #P = P[2, ...]
            #print(P.shape)
            C = P.shape[0]
            for c in range(C):
                P_ = P[c, ...] 
                cam = cv2.resize(P_, (width, height))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap / np.max(heatmap)
                heatmap_image = np.uint8(255 * heatmap)

                cv2.imwrite(P_save_path + '/'+ str(c)+ '_heatmap.jpg', heatmap_image)  # 生成图像
                result = cv2.addWeighted(image, 0.8, heatmap_image, 0.5, 0)
                cv2.imwrite(P_save_path + '/'+ str(c)+ '_result.jpg', result)


if __name__ == '__main__':
    main()
