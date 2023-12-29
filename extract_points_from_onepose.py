# %%
from pope_model_api import *
from pathlib import Path

# %%
ckpt, model_type = get_model_info("h")
sam = sam_model_registry[model_type](checkpoint=ckpt)
DEVICE = "cuda"
sam.to(device=DEVICE)
MASK_GEN = SamAutomaticMaskGenerator(sam)
logger.info(f"load SAM model from {ckpt}")
crop_tool = CropImage()
dinov2_model = load_dinov2_model()
dinov2_model.to("cuda:0")

# %%
metrics = dict()
metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})

# %%
id2name_dict = {
    1: "ape",
    2: "benchvise",
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

# %%
# load model
ROOT_DIR = "data/onepose/"
res_table = []

# %%
import json
with open("data/pairs/Onepose-test.json") as f:
    dir_list = json.load(f)

# %%
for label_idx, test_dict in enumerate(dir_list):
    logger.info(f"Onepose: {label_idx}")
    metrics = dict()
    metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})
    sample_data = dir_list[label_idx]["0"][0]
    label = sample_data.split("/")[0]
    name = label.split("-")[1]
    dir_name = os.path.dirname(sample_data)
    FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name)
    recall_image, all_image = 0, 0
    for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
        for pair_idx,pair_name in enumerate(tqdm(rotation_list)):
            all_image = all_image + 1
            base_name = os.path.basename(pair_name)
            idx0_name = base_name.split("-")[0]
            idx1_name = base_name.split("-")[1]
            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color"), idx1_name)
            intrinsic_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
            K0 = np.loadtxt(intrinsic_path, delimiter=' ')
            intrinsic_path = image1_name.replace("color", "intrin_ba").replace("png", "txt")
            K1 = np.loadtxt(intrinsic_path, delimiter=' ')
            image0 = cv2.imread(image0_name)
            ref_torch_image = set_torch_image(image0, center_crop=True)
            ref_fea = get_cls_token_torch(dinov2_model, ref_torch_image)
            image1 = cv2.imread(image1_name)
            image_h, image_w, _ = image1.shape
            t1 = time.time()
            masks = MASK_GEN.generate(image1)
            t2 = time.time()
            similarity_score, top_images  = np.array([0, 0, 0], np.float32), [[], [], []]
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
                # object_mask, _ = get_image_crop_resize(object_mask, box, resize_shape)
                box_new = np.array([0, 0, x1 - x0, y1 - y0])
                resize_shape = np.array([512, 512])
                K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
                image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
                crop_tensor = set_torch_image(image_crop, center_crop=True)
                with torch.no_grad():
                    fea = get_cls_token_torch(dinov2_model, crop_tensor)
                score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)
                if  (score.item() > similarity_score).any():
                    mask["crop_image"] = image_crop
                    mask["K"] = K_crop
                    mask["bbox"] = box
                    min_idx = np.argmin(similarity_score)
                    similarity_score[min_idx] = score.item()
                    top_images[min_idx] = mask.copy()

            img0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            img0 = torch.from_numpy(img0).float()[None] / 255.
            img0 = img0.unsqueeze(0).cuda()

            matching_score = [[0] for _ in range(len(top_images))]
            for top_idx in range(len(top_images)):
                if "crop_image" not in top_images[top_idx]:
                    continue
                img1 = cv2.cvtColor(top_images[top_idx]["crop_image"], cv2.COLOR_BGR2GRAY)
                img1 = torch.from_numpy(img1).float()[None] / 255.
                img1 = img1.unsqueeze(0).cuda()
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
            t4 = time.time()

            if "crop_image" not in top_images[top_idx]:
                continue
            max_match_idx = np.argmax(matching_score)
            pre_bbox = top_images[max_match_idx]["bbox"] # 需要保存
            mkpts0 = top_images[max_match_idx]["mkpts0"] # 需要保存
            mkpts1 = top_images[max_match_idx]["mkpts1"] # 需要保存
            pre_K = top_images[max_match_idx]["K"]       # 需要保存
            if (mkpts0.shape[0] < 5
                or mkpts1.shape[0] < 5
                or pre_K.shape[0] != 3):
                continue

            points_file_path = os.path.join("data/onepose-points/", pair_name.split("/")[0])
            pre_bbox_path = os.path.join(points_file_path, "pre_bbox")
            mkpts0_path = os.path.join(points_file_path, "mkpts0")
            mkpts1_path = os.path.join(points_file_path, "mkpts1")
            pre_K_path = os.path.join(points_file_path, "pre_K")
            Path(pre_bbox_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts0_path).mkdir(parents=True, exist_ok=True)
            Path(mkpts1_path).mkdir(parents=True, exist_ok=True)
            Path(pre_K_path).mkdir(parents=True, exist_ok=True)
            # print("points_file_path =", points_file_path)
            # print("mkpts0_path =", mkpts0_path)
            # print("mkpts1_path =", mkpts1_path)
            # print("pre_K_path =", pre_K_path)
            points_name = pair_name.split("/")[-1]

            np.savetxt(os.path.join(pre_bbox_path, f"{points_name}.txt"), pre_bbox)
            np.savetxt(os.path.join(mkpts0_path, f"{points_name}.txt"), mkpts0)
            np.savetxt(os.path.join(mkpts1_path, f"{points_name}.txt"), mkpts1)
            np.savetxt(os.path.join(pre_K_path, f"{points_name}.txt"), pre_K)

# %%



