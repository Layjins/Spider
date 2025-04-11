from StoryDiffusion.Comic_Generation import init_story_generation, story_generation

# models_dict = {
# "Juggernaut":"RunDiffusion/Juggernaut-XL-v8",
# "RealVision":"/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/RealVisXL_V4.0" ,
# "SDXL":"/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-xl-base-1.0" ,
# "Unstable": "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sdxl-unstable-diffusers-y"
# }
story_diffusion = init_story_generation(model_name="Unstable", device="cuda")


if __name__ == "__main__":
    # general_prompt = "a man with a black suit"
    # prompt_array = ["wake up in the bed",
    #                 "have breakfast",
    #                 "is on the road, go to the company",
    #                 "work in the company",
    #                 "running in the playground",
    #                 "reading book in the home"
    #                 ]
    # style_name = "Comic book" # "Japanese Anime", "Digital/Oil Painting", "Pixar/Disney Charactor", "Photographic", "Comic book", "Line art", "Black and White Film Noir", "Isometric Rooms"
    # preds = story_generation(story_diffusion, general_prompt=general_prompt, prompt_array=prompt_array, style_name=style_name)

    # story_diffusion = init_story_generation(model_name="Unstable", device="cuda")
    general_prompt = "a man with a black suit"
    prompt_array = ["wake up in the bed", "have breakfast", "work in the company", "reading book in the home"]
    style_name = "Comic book"
    preds = story_generation(story_diffusion, general_prompt=general_prompt, prompt_array=prompt_array, style_name=style_name)
