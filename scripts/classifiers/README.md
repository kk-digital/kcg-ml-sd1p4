# SD Chad - Stable Diffusion Aesthetic Scorer

Have been using SD to create art for the last month, finding a template that works across prompt, seed, settings, and then creating 100s of images from it, selecting the best, deleting the rest. 

That flow works great already, have lots of pics that look as good as those trending on ArtStation. Then I thought about automating this using AI. Here is what I have done so far: 

- Got the 10M prompts, seeds, settings, urls from the SD beta discord bot, Aug 2022 (same set used for krea.ai and lexica.art)
https://github.com/krea-ai/open-prompts
- Normalized the data by selecting 548K samples that are 512 x 512, 50 steps, 7 scale, not repeated seeds + prompt combo
https://drive.google.com/file/d/1c4WHxtlzvHYd0UY5WCMJNn2EO-Aiv2A0/view
- Downloaded the images, got image and text embeddings using img2dataset and Clip Retrieval (amazing framework)
https://github.com/rom1504/clip-retrieval
- Ran those images through ASV1 (created by LAION from 4K gens) and ASV2 (created from 250K gens + no gens, used to get the image set that SD was trained on)  
https://github.com/LAION-AI/aesthetic-predictor  
https://github.com/christophschuhmann/improved-aesthetic-predictor
- Created a script for SD, so one can generate images, and depending on score, they get saved in different folders, different file names
https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py
- Retrained the scoring model using 200K gens scored by people 1-10 to create a second comparing option (ASV2 = SD vs real images, CSV1 = SD vs SD)  
https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth
- Researched the correlation between scores at step 1 and step 20 across 200K gens, not repeated seeds + prompt combos (small 37% positive correlation, not usable)

Now I am retraining the scoring model again using the top 2,500 images scored from 200K gens (1.5 model + new VAE + style = best gens so far), and handpicking the few images that I would personally published in my art channels. First test was great, the model really seems to understand which images I would pick. 

# SD Chad Script

- Add the script to SD by following these instructions  
https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py   
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts
- Select the script from the dropdown, and add how many gens you want to produce (this adds all gens above 6.9 to your outputs folder, that's the statistically significant score limit from 500K scored gens; if you use the chadscorer model instead, change the limit to 7.7)   
https://github.com/grexzen/SD-Chad/blob/main/chadscorer.pth
- Create gens from lists of prompts and seeds by using this script in the custom script option on SD (create 1 txt for prompt + 1 for seeds, add paths to script)  
https://github.com/grexzen/SD-Chad/blob/main/create_gens_from_list_of_prompts_and_seeds_and_score_them.py 
- Retrain the scoring model by using the training scripts in this repo (create cvs with a list of images with 2 columns IMAGEPATH & AVERAGE_RATING, add paths to your images and scores, transform it into a parquet file, then run them).   
https://github.com/grexzen/SD-Chad/blob/main/simple_inference.py   
https://github.com/grexzen/SD-Chad/blob/main/train_predictor.py

![image](https://user-images.githubusercontent.com/30579087/199604457-0487daaa-0c9f-46c6-8eb8-59013d76652e.png)

# ASV1 vs ASV2

Here is ASV1. Album score =10 https://ibb.co/album/cY7GQW. Album score = 0 https://ibb.co/album/84p0Bk.

![image](https://user-images.githubusercontent.com/30579087/194484756-c7458d79-876f-4494-a431-604046efa26b.png)

Here is ASV2. Album score = 8 (highest) https://ibb.co/album/ypWyhL. Album score = 2 (lowest) https://ibb.co/album/0Rk3Yx.

![image](https://user-images.githubusercontent.com/30579087/194484962-d578229c-6c0c-4909-a98f-5fe7b59eb672.png)

ASV1 has a nicer distribution of scores, while ASV2 is pretty tight in the middle. Since ASV2 was created by scoring non-gens that might be why is so strict scoring gens from SD. It also seems to prefer realistic images.

![image](https://user-images.githubusercontent.com/30579087/194380290-6c68fb6b-cd78-4d9f-9ff2-bdbe2c3d43bb.png)

# ASV2 vs CSV1

ASV2 is great for scoring your images vs other gens + no gens (SD vs real pics) and CSV1 is great for scoring your images vs other gens only (SD vs SD).

Below are the distributions, average scores, and 2 & 3 standard deviations from the mean.

![image](https://user-images.githubusercontent.com/30579087/199601438-0f7b4860-aad5-4204-97fb-c7cf20c69c44.png)
