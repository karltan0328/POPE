# %%
from pope_model_api import *
from pathlib import Path

# %%
ckpt, model_type = get_model_info("h")
sam = sam_model_registry[model_type](checkpoint=ckpt).to('cuda:3')
MASK_GEN = SamAutomaticMaskGenerator(sam)
logger.info(f"load SAM model from {ckpt}")
crop_tool = CropImage()
dinov2_model = load_dinov2_model().to('cuda:1')

# %%
metrics = dict()
metrics.update({'R_errs': [], 't_errs': [], 'inliers': [] , "identifiers":[] })


# load data ROOR_DIR
ROOT_DIR = "data/ycbv/"
res_table = []

import json
with open("data/pairs/YCB-VIDEO-test.json") as f:
    dir_list = json.load(f)

# %%
for label_idx, test_dict in enumerate(dir_list):
    logger.info(f"YCBVIDEO: {label_idx}")
    metrics = dict()
    metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})
    sample_data = dir_list[label_idx]["0"][0]
    label = sample_data.split("/")[0]
    name = label.split("-")[1]
    dir_name = os.path.dirname(sample_data)
    FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name)
    recall_image, all_image = 0, 0
    for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
        for pair_idx, pair_name in enumerate(tqdm(rotation_list)):
            all_image = all_image + 1
            base_name = os.path.basename(pair_name)
            idx0_name = base_name.split("png-")[0] + "png"
            idx1_name = base_name.split("png-")[1]
            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color_full"), idx1_name)
            intrinsic_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")

            try:
                K0 = np.loadtxt(intrinsic_path, delimiter=' ')
            except:
                # print(f'{intrinsic_path} is not exist')
                continue
            intrinsic_path = image1_name.replace("color_full", "intrin").replace("png", "txt")
            try:
                K1 = np.loadtxt(intrinsic_path, delimiter=' ')
            except:
                # print(f'{intrinsic_path} is not exist')
                continue
            image0 = cv2.imread(image0_name)
            ref_torch_image = set_torch_image(image0, center_crop=True)
            ref_fea = get_cls_token_torch(dinov2_model, ref_torch_image)
            image1 = cv2.imread(image1_name)
            image_h, image_w, _ = image1.shape
            t1 = time.time()
            masks = MASK_GEN.generate(image1)
            t2 = time.time()
            similarity_score, top_images = np.array([0, 0, 0], np.float32), [[], [], []]
            t3 = time.time()
            compact_percent = 0.3
            for xxx, mask in enumerate(masks):
                object_mask = np.expand_dims(mask["segmentation"], -1)
                x0, y0, w, h = mask["bbox"]
                x1, y1 = x0 + w, y0 + h
                x0 -= int(w * compact_percent)
                y0 -= int(h * compact_percent)
                x1 += int(w * compact_percent)
                y1 += int(h * compact_percent)
                box = np.array([x0, y0, x1, y1])
                resize_shape = np.array([y1 - y0, x1 - x0])
                K_crop, K_crop_homo = get_K_crop_resize(box, K1, resize_shape)
                image_crop, _ = get_image_crop_resize(image1, box, resize_shape)
                # object_mask,_ = get_image_crop_resize(object_mask, box, resize_shape)
                box_new = np.array([0, 0, x1 - x0, y1 - y0])
                resize_shape = np.array([256, 256])
                K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
                image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
                crop_tensor = set_torch_image(image_crop, center_crop=True)
                with torch.no_grad():
                    fea = get_cls_token_torch(dinov2_model, crop_tensor)
                score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)
                if (score.item() > similarity_score).any():
                    mask["crop_image"] = image_crop
                    mask["K"] = K_crop
                    mask["bbox"] = box
                    min_idx = np.argmin(similarity_score)
                    similarity_score[min_idx] = score.item()
                    top_images[min_idx] = mask.copy()

            crop_img0 = image0
            img0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            img0 = torch.from_numpy(img0).float()[None] / 255.
            img0 = img0.unsqueeze(0).to('cuda:1')

            matching_score = [[0] for _ in range(len(top_images))]
            for top_idx in range(len(top_images)):
                crop_img1 = top_images[top_idx]["crop_image"]
                img1 = cv2.cvtColor(top_images[top_idx]["crop_image"], cv2.COLOR_BGR2GRAY)
                img1 = torch.from_numpy(img1).float()[None] / 255.
                img1 = img1.unsqueeze(0).to('cuda:1')
                batch = {'image0':img0, 'image1':img1}
                with torch.no_grad():
                    matcher(batch)
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()
                    confidences = batch["mconf"].cpu().numpy()
                conf_mask = np.where(confidences > 0.9)
                matching_score[top_idx] = conf_mask[0].shape[0]
                top_images[top_idx]["mkpts0"] = mkpts0
                top_images[top_idx]["mkpts1"] = mkpts1
                top_images[top_idx]["mconf"] = confidences
                top_images[top_idx]["crop_img0"] = crop_img0
                top_images[top_idx]["crop_img1"] = crop_img1
            #---------------------------------------------------
            # crop_image = cv2.resize(top_images[np.argmax(matching_score)]["crop_image"],(256,256))
            # que_image = cv2.resize(image0,(256,256))
            # image = np.hstack((que_image, crop_image))
            # for top_idx in range(len(top_images)):
            #     crop_image = top_images[top_idx]["crop_image"]
            #     score = matching_score[top_idx]
            #     crop_image = cv2.resize(crop_image,(256,256))
            #     cv2.putText(crop_image,f'{score}',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            #     image = np.hstack((image, crop_image))
            # cv2.imwrite(f"segment_anything/crop_images/{idx}.jpg", image)
            #---------------------------------------------------
            t4 = time.time()
            # print(f"t4-t3: object detection:{1000*(t4-t3)} ms")
            pose0_name = image0_name.replace("color", "poses_ba").replace("png", "txt")
            pose1_name = image1_name.replace("color_full", "poses_ba").replace("png", "txt")
            pose0 = np.loadtxt(pose0_name)
            pose1 = np.loadtxt(pose1_name)
            pose0 = np.vstack((pose0, np.array([[0, 0, 0, 1]])))
            pose1 = np.vstack((pose1, np.array([[0, 0, 0, 1]])))
            relative_pose =  np.matmul(pose1, inv(pose0))
            t = relative_pose[:3, -1].reshape(1, 3)

            max_match_idx = np.argmax(matching_score)
            pre_bbox  = top_images[max_match_idx]["bbox"]
            mkpts0 = top_images[max_match_idx]["mkpts0"]
            mkpts1 = top_images[max_match_idx]["mkpts1"]
            pre_K = top_images[max_match_idx]["K"]
            crop_img0 = top_images[max_match_idx]["crop_img0"]
            crop_img1 = top_images[max_match_idx]["crop_img1"]
            if (mkpts0.shape[0] < 5
                or mkpts1.shape[0] < 5
                or pre_K.shape[0] != 3):
                continue

            points_file_path = os.path.join("data/ycbv-points/", pair_name.split("/")[0])
            pre_bbox_path = os.path.join(points_file_path, "pre_bbox")
            mkpts0_path = os.path.join(points_file_path, "mkpts0")
            mkpts1_path = os.path.join(points_file_path, "mkpts1")
            pre_K_path = os.path.join(points_file_path, "pre_K")
            crop_img0_path = os.path.join(points_file_path, "img0")
            crop_img1_path = os.path.join(points_file_path, "img1")
            Path(pre_bbox_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts0_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts1_path).mkdir(parents=True, exist_ok=True)
            Path(pre_K_path).mkdir(parents=True, exist_ok=True)
            Path(crop_img0_path).mkdir(parents=True, exist_ok=True)
            Path(crop_img1_path).mkdir(parents=True, exist_ok=True)
            # print("points_file_path =", points_file_path)
            # print("mkpts0_path =", mkpts0_path)
            # print("mkpts1_path =", mkpts1_path)
            # print("pre_K_path =", pre_K_path)
            points_name = pair_name.split("/")[-1]

            np.savetxt(os.path.join(pre_bbox_path, f"{points_name}.txt"), pre_bbox)
            np.savetxt(os.path.join(mkpts0_path, f"{points_name}.txt"), mkpts0)
            np.savetxt(os.path.join(mkpts1_path, f"{points_name}.txt"), mkpts1)
            np.savetxt(os.path.join(pre_K_path, f"{points_name}.txt"), pre_K)
            cv2.imwrite(os.path.join(crop_img0_path, f"{points_name}.png"), crop_img0)
            cv2.imwrite(os.path.join(crop_img1_path, f"{points_name}.png"), crop_img1)

# %%



