from mmdet.apis import init_detector, inference_detector
from transformers import logging
logging.set_verbosity_error()

config_file = '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
checkpoint_file = '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0' or device='cpu'
res = inference_detector(model, '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/apple.jpg', text_prompt='apple')
print(res)

import imageio.v3 as iio
image_iio = iio.imread('/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/apple.jpg', pilmode="RGB")
res_image_iio = inference_detector(model, image_iio, text_prompt='apple')
print(res_image_iio)


'''
(Pdb) res.keys()
['pred_instances', 'token_positive_map', 'gt_instances', 'ignored_instances']

(Pdb) res.pred_instances.bboxes.shape
torch.Size([300, 4])

(Pdb) res.pred_instances.bboxes
tensor([[ 187.3906,  255.4016, 1066.5681, 1093.0085],
        [ 186.3487,   69.7367, 1067.3656, 1094.3219],
        [ 467.8506,  262.4486,  721.6009,  413.4679],
        ...,
        [ 796.8724,  982.6337, 1053.4812, 1125.0996],
        [1155.0032,  322.5294, 1278.7494, 1120.4579],
        [ 860.3583,  753.2054,  966.2690,  850.0090]], device='cuda:0')

(Pdb) res.pred_instances
<InstanceData(

    META INFORMATION

    DATA FIELDS
    labels: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
    label_names: ['apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple']
    bboxes: tensor([[ 187.3906,  255.4016, 1066.5681, 1093.0085],
                [ 186.3487,   69.7367, 1067.3656, 1094.3219],
                [ 467.8506,  262.4486,  721.6009,  413.4679],
                ...,
                [ 796.8724,  982.6337, 1053.4812, 1125.0996],
                [1155.0032,  322.5294, 1278.7494, 1120.4579],
                [ 860.3583,  753.2054,  966.2690,  850.0090]], device='cuda:0')
    scores: tensor([1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 9.9999e-01,
                9.9987e-01, 9.9978e-01, 9.9954e-01, 9.9932e-01, 9.9876e-01, 9.9642e-01,
                9.8307e-01, 9.7499e-01, 9.3486e-01, 7.4474e-01, 7.0234e-01, 5.6033e-01,
                4.3614e-01, 2.0972e-01, 1.3004e-01, 1.1381e-01, 1.0719e-01, 8.6759e-02,
                7.5940e-02, 7.2066e-02, 4.3733e-02, 4.3686e-02, 2.9397e-02, 2.2729e-02,
                1.7015e-02, 1.5885e-02, 1.4844e-02, 1.4353e-02, 1.1468e-02, 8.7818e-03,
                3.8738e-03, 3.4058e-03, 3.0853e-03, 3.0725e-03, 1.1438e-03, 1.0633e-03,
                1.0061e-03, 8.4526e-04, 8.2618e-04, 6.4587e-04, 6.3925e-04, 6.1105e-04,
                5.9633e-04, 5.2872e-04, 4.7969e-04, 4.6816e-04, 4.5045e-04, 3.5832e-04,
                3.1780e-04, 2.9870e-04, 2.9551e-04, 2.1769e-04, 1.7918e-04, 1.7748e-04,
                1.7595e-04, 1.5321e-04, 1.4944e-04, 1.4799e-04, 1.2988e-04, 1.2831e-04,
                1.1920e-04, 1.1154e-04, 1.0274e-04, 7.5123e-05, 7.2868e-05, 5.9806e-05,
                5.2354e-05, 4.8986e-05, 3.7021e-05, 3.4075e-05, 3.0792e-05, 2.6067e-05,
                2.3810e-05, 2.3198e-05, 1.9826e-05, 1.8519e-05, 1.6886e-05, 1.6109e-05,
                1.1358e-05, 9.7419e-06, 9.5024e-06, 9.3698e-06, 8.9449e-06, 8.6080e-06,
                7.8032e-06, 6.9776e-06, 6.7834e-06, 6.5287e-06, 6.0969e-06, 5.9297e-06,
                5.7167e-06, 4.9816e-06, 4.7980e-06, 4.2394e-06, 4.1977e-06, 3.8619e-06,
                3.3753e-06, 3.0232e-06, 2.9559e-06, 2.6650e-06, 2.6046e-06, 1.7488e-06,
                1.7376e-06, 1.7061e-06, 1.6507e-06, 1.4853e-06, 1.3868e-06, 1.3000e-06,
                1.2812e-06, 1.0787e-06, 9.7819e-07, 9.3700e-07, 9.2096e-07, 8.4745e-07,
                7.9375e-07, 7.9007e-07, 7.0643e-07, 7.0422e-07, 6.8763e-07, 6.6586e-07,
                5.4097e-07, 5.2332e-07, 5.0442e-07, 3.8657e-07, 3.3079e-07, 3.1307e-07,
                2.7582e-07, 2.7489e-07, 2.4720e-07, 2.3179e-07, 2.1327e-07, 2.0760e-07,
                2.0449e-07, 1.8406e-07, 1.6045e-07, 1.5516e-07, 1.5241e-07, 1.3474e-07,
                1.3281e-07, 1.1640e-07, 1.0066e-07, 1.0037e-07, 9.3268e-08, 5.4836e-08,
                5.4331e-08, 5.2587e-08, 5.1147e-08, 5.0148e-08, 4.8464e-08, 4.6453e-08,
                4.2505e-08, 3.9152e-08, 3.7671e-08, 3.5445e-08, 3.0396e-08, 2.8805e-08,
                2.8514e-08, 2.7626e-08, 2.7423e-08, 2.6498e-08, 2.2653e-08, 2.2055e-08,
                2.1761e-08, 2.0853e-08, 2.0766e-08, 1.8923e-08, 1.8915e-08, 1.8853e-08,
                1.5805e-08, 1.5510e-08, 1.3181e-08, 1.1379e-08, 1.1212e-08, 1.0645e-08,
                1.0188e-08, 9.9184e-09, 9.6500e-09, 9.3483e-09, 8.2174e-09, 7.9649e-09,
                7.2722e-09, 7.1006e-09, 6.9394e-09, 6.6044e-09, 6.5893e-09, 6.5369e-09,
                5.8834e-09, 5.7404e-09, 5.4882e-09, 5.4187e-09, 5.3823e-09, 5.3686e-09,
                5.3393e-09, 3.7702e-09, 3.7586e-09, 3.6335e-09, 3.5756e-09, 3.4241e-09,
                3.1371e-09, 3.0655e-09, 2.8996e-09, 2.8936e-09, 2.7930e-09, 2.7839e-09,
                2.7264e-09, 2.4407e-09, 2.4005e-09, 2.3510e-09, 2.2059e-09, 2.0282e-09,
                1.8979e-09, 1.8780e-09, 1.8719e-09, 1.8593e-09, 1.8093e-09, 1.8062e-09,
                1.6990e-09, 1.6435e-09, 1.6434e-09, 1.6147e-09, 1.5052e-09, 1.4809e-09,
                1.3433e-09, 1.3118e-09, 1.2409e-09, 1.1472e-09, 1.0286e-09, 9.4805e-10,
                8.9811e-10, 8.8512e-10, 8.8452e-10, 7.4987e-10, 7.3574e-10, 7.0885e-10,
                6.5069e-10, 6.0210e-10, 5.9929e-10, 5.8744e-10, 4.7781e-10, 4.7458e-10,
                4.4282e-10, 4.4216e-10, 4.3282e-10, 4.3083e-10, 4.3058e-10, 4.1347e-10,
                3.9343e-10, 3.8119e-10, 3.7873e-10, 3.7490e-10, 3.4562e-10, 3.3307e-10,
                3.0823e-10, 2.8733e-10, 2.5252e-10, 2.4879e-10, 2.1534e-10, 2.1153e-10,
                1.9392e-10, 1.9044e-10, 1.8981e-10, 1.8981e-10, 1.8764e-10, 1.7690e-10,
                1.7392e-10, 1.6769e-10, 1.6350e-10, 1.6181e-10, 1.5535e-10, 1.5242e-10,
                1.5126e-10, 1.4961e-10, 1.4304e-10, 1.4171e-10, 1.3666e-10, 1.3168e-10,
                1.2856e-10, 1.2697e-10, 1.2464e-10, 1.2419e-10, 1.1811e-10, 1.0022e-10,
                9.8700e-11, 9.2601e-11, 9.1684e-11, 8.5591e-11, 8.1356e-11, 7.8711e-11,
                7.6897e-11, 7.4762e-11, 7.1692e-11, 7.0113e-11, 6.3356e-11, 6.2138e-11],
               device='cuda:0')
)
'''