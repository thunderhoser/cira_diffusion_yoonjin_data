from diffusers.pipelines import DiffusionPipeline,ImagePipelineOutput
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
class DDPMCondPipeline2(DiffusionPipeline):
    r"""
    This was a pieced together pipeline to take a condition. 

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        conditioning,  # TODO: Set this as a numpy array type or Torch tensor, expected to be [batch,channel,x,y]
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            conditioning (`torch.FloatTensor`):
                The conditioning info to pass into the UNet.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Calculate batch size
        batch_size = conditioning.shape[0]


        #hard code shape for now 
        image_shape = (batch_size,
                1,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        
        # Sample gaussian noise to begin loop
        # if isinstance(self.unet.config.sample_size, int):
        #     image_shape = (
        #         batch_size,
        #         self.unet.config.in_channels,
        #         self.unet.config.sample_size,
        #         self.unet.config.sample_size,
        #     )
        # else:
        #     image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        #make sure condition is on the same device
        conditioning = conditioning.to(self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            #1 add condition to the input 
            model_input_images = torch.concatenate([image,conditioning],axis=1)
            
            #2 predict noise model_output
            model_output = self.unet(model_input_images, t).sample

            #3 compute previous image: x_t -> x_t-1 from just the noisy image and the model output
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        # if output_type == "pil":
            # image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)