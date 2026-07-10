# Octave UAED implementation
To acess the original implementation, acess https://github.com/ZhouCX117/UAED_MuGE


# UAED
The Treasure Beneath Multiple Annotations: An Uncertainty-aware Edge Detector  
Caixia Zhou, Yaping Huang, Mengyang Pu, Qingji Guan, Li Huang and Haibin Ling  
CVPR 2023

# Preparing Data
The processed dataset is from LPCB, you can download the used matlab code and processed data from the [Baidu disk](https://pan.baidu.com/s/1F2nAYKsmNxTCI6dmAOGQqg), the code is 3tii. Or downloaded in the drive https://drive.google.com/drive/folders/1VcO4dnEVRsSBdTxBN0itiLKNQ_kUbxsv?usp=drive_link
The complete processed BSDS training dataset can be downloaded from the [Google disk](https://drive.google.com/file/d/1iB2aUKTjDK0URbvUXbXBKBYAROftRKwX/view?usp=sharing).

# Checkpoint 
BSDS with single scale for UAED: [Quark disk](https://pan.quark.cn/s/9e65e82b3d40) or  [Google disk](https://drive.google.com/file/d/1nv2_TZRyiQh5oU9TnGMzu313OrspD2l5/view?usp=sharing)  
  
# Results
UAED Results for BSDS under a single-scale setting can be found [here](https://pan.quark.cn/s/840cd0690997).
# Start
UAED:  
```
python train_uaed.py
```
* `--batch_size` : Specifies the number of images processed in each training batch. Default value: 4.
* `--LR / --learning_rate` : Defines the initial learning rate used by the Adam optimizer. 
Controls the magnitude of weight updates during training.
Default value: 0.0001.
* `--weight_decay / --wd` : Specifies the L2 regularization factor applied to the model weights. 
Helps reduce overfitting by penalizing large weights.
Default value: 0.0005.
* `--stepsize` : Defines the number of epochs between learning rate updates. 
Every stepsize epochs, the learning rate is multiplied by 0.1.
Default value: 3.
* `--maxepoch` : Specifies the total number of training epochs. 
Default value: 20.
`*--start_epoch` : Defines the starting epoch of the training process.
Useful when resuming training from a saved checkpoint.
Default value: 0.
* `--print_freq (-p)` : Determines how often training statistics (e.g., loss and execution time) are printed during training.
Default value: 1000 iterations.
* `--gpu` : Specifies the CUDA GPU ID used for training.
Example: "0" selects the first available GPU.
Default value: "0".
* `--tmp` : Specifies the directory where logs, checkpoints, and intermediate results are stored.
Default value: ./temp/train.
* `--dataset` : Defines the root directory containing the training and testing datasets.
Default value: ./datasets/BSDS/.
* `--itersize` : Specifies the number of iterations over which gradients are accumulated before updating the model parameters.
Allows the simulation of larger batch sizes when GPU memory is limited.
Default value: 1.
* `--std_weight` : Defines the weight assigned to the standard deviation loss term.
Controls the contribution of uncertainty estimation to the overall loss function.
Default value: 1.
* `--distribution` : Specifies the output probability distribution assumed by the model.
The default implementation uses "gs".
Default value: "gs".
* `--scale_test` : Enables or disables multi-scale inference during testing.
When enabled, predictions from multiple image scales are averaged to improve performance.
Default value: False.
* `--attention` : Specifies the attention mechanism used by the model.
According to the implementation, the supported option is "excitation".
Default value: None.
* `--model_file` : Specifies the Python module containing the neural network architecture to be trained.
Allows different model implementations to be selected without modifying the training script.
Default value: model.sigma_logit_unetpp.


# Original implementation and paper
The dataset is highly based on the LPCB, and the code is highly based on [RCF_Pytorch_Updated](https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated) and [
segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). Many thanks for their great work.  
Please consider citing this project in your publications if it helps your research.
```
@inproceedings{zhou2023treasure,
  title={The treasure beneath multiple annotations: An uncertainty-aware edge detector},
  author={Zhou, Caixia and Huang, Yaping and Pu, Mengyang and Guan, Qingji and Huang, Li and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15507--15517},
  year={2023}
}
```
```
@inproceedings{zhou2024muge,
  title={MuGE: Multiple Granularity Edge Detection},
  author={Zhou, Caixia and Huang, Yaping and Pu, Mengyang and Guan, Qingji and Deng, Ruoxi and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
```
@inproceedings{deng2018learning,
  title={Learning to predict crisp boundaries},
  author={Deng, Ruoxi and Shen, Chunhua and Liu, Shengjun and Wang, Huibing and Liu, Xinru},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={562--578},
  year={2018}
}
```

