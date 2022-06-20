# Structure_perserving_NST
Neural style transfer with a complex loss function which comprehend depth, edge, style and content.
This notebook aims to integrate the modifications presented by by Cheng et al. in the paper entitled ['Structure perserving Neural Style Transfer'](https://www.researchgate.net/publication/335441628_Structure-Preserving_Neural_Style_Transfer)  to preserve structure on Neural Style Transfer. These used technique for NST is Gatys Model in ['A Neural Algorithm of Artistic Style'](https://arxiv.org/abs/1508.06576), which is basically an IOB (Image optimization based) method. 

The first paper presents a methodology to preserve the semantic integrity of the image uncompromised by the strokes of the brush in the generated pastiche, which can be accomplished by adding edge loss and depth loss in addition to the original content, style and variation losses. This notebook aims to test the modified loss on the original methodology for NST by [Gatys et al](https://arxiv.org/abs/1508.06576).

As Gatys's model presents a benchmark optimization technique, and often generates the optimal quality pastiches [1](https://arxiv.org/abs/1705.04058), it implies a slow iterative process in order to generate the pastiche. The methodology is to start with an initialization for the generated image, either with a white noise or a clone of the content, and  calculate the content loss and the style loss using pictures representations using VGG19, then backpropagate over the generated image gradually to generate a pastiche with minimal loss.

The pastiche generation process is optimized based on the overall loss which is a weighted summation of content loss and style loss. Content loss is calculated by Euclidean distance between content and target images, while style loss is calculated by the Euclidean distance between the gram matrix of content and target images. This notebook would aim to fine-tuning Gatys's methodology on a more complex loss function, which will include edge loss and depth loss, which would be calculated as the Euclidian distance between pastiche representation and content representation through a global structure detector module using MiDaS depth estimation model which has been presented in [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341),  and a local structure detector module using the model introduced in ["Holistically-Nested Edge Detection"](https://arxiv.org/abs/1504.06375) by Xie et al.

## Notebook Structure

0- Importing the necessary libraries and frameworks\
1- Defining Hyper-parameters and Helper Functions\
2- Creation of Data Object\
3- Creation of Data Loader\
4- Importing the models used in Loss Calculation
5- Defining Loss Functions\
6- Image Optimization Class with Multisize Stroke Feature\
7- Testing the Optimization Algorithm\
8- Visualizing & Comparing the Different Pastiches\
9- Fine Tuning Optimization\
10- Creating a Video Recorder\
11- References

## Note
The original paper of Structure perserving Neural Style Transfer utilized the model presented in ["Single-image depth perception
in the wild"](https://arxiv.org/abs/1604.03901) by Chen et al. instead of MiDaS model for depth estimation, while I used MiDaS because of its ubiquity.

## Credits for different implementations utilized in this notebook
#### Papers: 

1. [Neural Style Transfer Review](https://github.com/ycjing/Neural-Style-Transfer-Papers)
2. [Gatys's Paper](https://arxiv.org/abs/1508.06576)
3. [Ghiasi's Paper](https://arxiv.org/pdf/1705.06830.pdf)
4. [Structure poerserving Neural Style Transfer](https://www.researchgate.net/publication/335441628_Structure-Preserving_Neural_Style_Transfer)
5. [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375)
6. [Single-Image Depth Perception in the Wild](https://proceedings.neurips.cc/paper/2016/file/0deb1c54814305ca9ad266f53bc82511-Paper.pdf)


#### Implementations

1. [TensorFlow Lite Tutorial](https://www.tensorflow.org/lite/examples/style_transfer/overview)
2. [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
3. [PyTorch Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html?highlight=neural%20style)

4. [Magenta](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization)
5. [Implementations for Ghiasi Model](https://paperswithcode.com/paper/exploring-the-structure-of-a-real-time)
6. [Real time style based implementation](https://medium.com/@chimezie.iwuanyanwu/real-time-style-transfer-caffa3393833)
7. [Structure-Preserving Neural Style Transfer](https://github.com/xch-liu/structure-nst)
8. [Structure Preserving Implementation](https://github.com/xch-liu/structure-nst)
9. [Total Variation Loss](https://towardsdatascience.com/practical-techniques-for-getting-style-transfer-to-work-19884a0d69eb#:~:text=Total%20variation%20loss%20is%20the,noise%20is%20in%20the%20images.)
10. [Useful Experiments on NST](https://towardsdatascience.com/experiments-on-different-loss-configurations-for-style-transfer-7e3147eda55e)
11. [NST Implementation](https://github.com/ProGamerGov/neural-style-pt)
12. [HED Model on Github](https://github.com/meteorshowers/hed)
13. [Depth Model on Github](https://github.com/xch-liu/structure-nst/blob/master/doc/training.md)
