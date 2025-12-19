# VLM

## 3D Scene Understanding

### Data prepration

Download json files from [here](https://huggingface.co/AIDC-AI/Omni-View/tree/main/eval_dataset). Move the json files to `./dataset/eval/3dscene/` 

Download metadata from [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data). You need to fill out the official form to get the access to the dataset. Move the `embodiedscan_infos_*.pkl` to `./dataset/eval/embodiedscan`.

Download images from [Video3DLLM](https://huggingface.co/datasets/zd11024/Video-3D-LLM_data). Then, 

```shell
cd Video-3D-LLM_data
# unzip posed images
cat posed_images_part* > posed_images.tar.gz
tar -xzf posed_images.tar.gz
# unzip mask
unzip mask.zip
# unzip pcd
tar -xzf pcd_with_object_aabbs.tar.gz

mkdir scannet
mv posed_images/ scannet/
mv mask/ scannet/
mv data/scannet/pcd_with_object_aabbs/ scannet/
```

Move the `scannet` to `./dataset/eval/`.

The whole file structure under `./dataset/eval/` will be as follows.

```shell
./dataset/eval/
├── 3dscene
│   ├── scannet_det_val_4frames.json
│   ├── scannet_select_frames.json
│   ├── scanqa_val_llava_style.json
│   ├── scanrefer_val_32frames.json
│   └── sqa3d_test_llava_style.json
├── embodiedscan
│   ├── embodiedscan_infos_test.pkl
│   ├── embodiedscan_infos_train.pkl
│   └── embodiedscan_infos_val.pkl
└── scannet
    ├── mask
    ├── pcd_with_object_aabbs
    └── posed_images

6 directories, 9 files
```

### Inference

```shell
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_3dvqa --model-path ./pretrained_model/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset sqa3d
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_3dvqa --model-path ./pretrained_model/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset scanqa
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_3dvqa --model-path ./pretrained_model/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset scanrefer
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_3dvqa --model-path ./pretrained_model/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset 3ddet
```

The results (`*.json` files) will be saved in `./results/`.

### Evaluation

#### SQA3D

```shell
python eval/vlm/eval/vqa/3dvqa_eval.py --dataset sqa3d --input-file ./results/sqa3d.json

# output
EM-all: 59.16453537936914
EM-what: 51.787271142109844
EM-which: 50.427350427350426
EM-can: 68.04733727810651
EM-is: 73.15950920245399
EM-how: 60.86021505376345
EM-others: 56.71378091872792

EM-R-all: 62.432509235578294
EM-R-what: 58.326068003487364
EM-R-which: 51.566951566951566
EM-R-can: 68.04733727810651
EM-R-is: 75.61349693251533
EM-R-how: 61.29032258064516
EM-R-others: 59.8939929328622
```

#### ScanQA

```shell
python eval/vlm/eval/vqa/3dvqa_eval.py --dataset scanqa --input-file ./results/scanqa.json

# output
CIDER: 103.01606173581641
BLEU: 48.355630543562654, 32.50084859567164, 23.329984202585262, 16.194281050888
METEOR: 20.101447665118403
Rouge: 49.04130603336627
EM: 29.497326203208555
```

#### ScanRefer

```shell
python eval/vlm/eval/vqa/3dgrounding_eval.py --input-file ./results/scanrefer.json

# output
all iou@0.25: 50.82036180058898
all iou@0.5: 45.0462768195204
multiple iou@0.25: 44.877985123319846
multiple iou@0.5: 39.34490408456218
unique iou@0.25: 75.50135501355012
unique iou@0.5: 68.72628726287263
```

#### 3D Detection

```shell
python eval/vlm/eval/vqa/3ddet_eval.py --input-file ./results/3ddet.json

# output
+Metrics Per Category------+--------+----------+
| Category     | Precision | Recall | F1 Score |
+--------------+-----------+--------+----------+
| chair        | 0.4807    | 0.5074 | 0.4937   |
| pillow       | 0.1683    | 0.1881 | 0.1777   |
| cabinet      | 0.1401    | 0.1608 | 0.1497   |
| table        | 0.3744    | 0.3528 | 0.3633   |
| lamp         | 0.1302    | 0.0986 | 0.1122   |
| couch        | 0.4795    | 0.4862 | 0.4828   |
| desk         | 0.3567    | 0.3896 | 0.3724   |
| stand        | 0.4167    | 0.3509 | 0.3810   |
| bed          | 0.7468    | 0.6797 | 0.7117   |
| backpack     | 0.3130    | 0.3628 | 0.3361   |
| bathtub      | 0.4343    | 0.4000 | 0.4164   |
| ottoman      | 0.1250    | 0.0714 | 0.0909   |
| dresser      | 0.4828    | 0.3825 | 0.4268   |
| bin          | 0.3727    | 0.3162 | 0.3421   |
| toilet       | 0.7720    | 0.7395 | 0.7554   |
| refrigerator | 0.3486    | 0.4176 | 0.3800   |
| stove        | 0.7826    | 0.7347 | 0.7579   |
| microwave    | 0.2453    | 0.1884 | 0.2131   |
| monitor      | 0.2422    | 0.2770 | 0.2585   |
| computer     | 0.1546    | 0.0968 | 0.1190   |
| window       | 0.1297    | 0.0997 | 0.1127   |
| shelf        | 0.1939    | 0.2184 | 0.2054   |
| curtain      | 0.1260    | 0.1291 | 0.1275   |
| plant        | 0.1538    | 0.0855 | 0.1099   |
| stairs       | 0.3243    | 0.4000 | 0.3582   |
| picture      | 0.0212    | 0.0212 | 0.0212   |
| book         | 0.0348    | 0.0629 | 0.0448   |
| bottle       | 0.0247    | 0.0284 | 0.0264   |
| lamp         | 0.1302    | 0.0986 | 0.1122   |
| towl         | 0.0000    | 0.0000 | 0.0000   |
| sink         | 0.4752    | 0.4467 | 0.4605   |
+--------------+-----------+--------+----------+
+--------+---------------+------------+--------+
| Split  | Avg Precision | Avg Recall | Avg F1 |
+--------+---------------+------------+--------+
| cate8  | 0.4751        | 0.4553     | 0.4644 |
| cate20 | 0.3783        | 0.3601     | 0.3670 |
| cate31 | 0.2961        | 0.2836     | 0.2877 |
+--------+---------------+------------+--------+
```

> During evaluation, the error `ERROR    | __main__:threedod_process_results:357 - Error parsing prediction bbox` may appear. This error does not affect the evaluation.



## VSI-Bench

### Data prepration

Download [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), put it in `dataset/eval`.

### Inference

```shell
torchrun --nproc_per_node=4 --master_port=12345 -m eval.vlm.eval.vqa.evaluate_vsibench --model-path ./pretrained_model/BAGEL-7B-MoT/ --safetensor-path model.safetensors --dataset vsibench
```

The results (`*.json` files) will be saved in `./results/`.

### Evaluation

```shell
python eval/vlm/eval/vqa/3dvqa_eval.py --dataset vsibench --input-file ./results/vsibench.json

# output
obj_appearance_order_accuracy: 49.029126213592235
object_abs_distance_MRA:.5:.95:.05: 46.402877697841724
object_counting_MRA:.5:.95:.05: 70.31858407079646
object_rel_distance_accuracy: 65.91549295774648
object_size_estimation_MRA:.5:.95:.05: 68.59391395592864
room_size_estimation_MRA:.5:.95:.05: 54.722222222222214
route_planning_accuracy: 33.50515463917525
object_rel_direction_accuracy: 54.404572390100945
overall: 55.3614930184255
```

## Novel View Synthesis

### Data prepration

Download the RealEstate10K dataset from [this link](http://schadenfreude.csail.mit.edu:8000/), which is provided by [pixelSplat](https://github.com/dcharatan/pixelsplat), and `unzip` the zip file and put the data in `YOUR_RAW_DATAPATH`.

Run the following command to preprocess the data into our format.

```shell
git clone https://github.com/zalkklop/LVSM.git
cd LVSM
python process_data.py --base_path YOUR_RAW_DATAPATH --output_dir ./dataset/eval/re10k/ --mode ['train' or 'test']
```

The whole file structure under `./dataset/eval/re10k/test/` will be as follows.

```
./dataset/eval/re10k/test/
├── full_list.txt
├── images
│   ├── 000c3ab189999a83
│   ├── ...
├── metadata
│   ├── 000c3ab189999a83.json
│   ├── ...
```

### Evaluation

We provide a script to evaluate Omni-View on [RE10k](https://google.github.io/realestate10k/).

```shell
python inference.py --scene-id 000c3ab189999a83
```

| Argument                     | Default                | Description                                                      |
| ---------------------------- | ---------------------- | ---------------------------------------------------------------- |
| `scene-id`                   | None                   | The scene id in RE10k.     |
| `pose-id`                    | None                   | The id of camera trajectory in RE10k. Default: pose_id = scene_id |
| `image-path`                 | None                   | The reference image path.  |

If `scene-id != pose-id`, we will use the first image of scene-id as the reference image and generate novel views using the camera trajectory of pose-id.

If `(scene-id is None) and (image-path is not None)`, we will use the image in image-path as the reference image and generate novel views using the camera trajectory of pose-id.
