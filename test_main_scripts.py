import subprocess
import unittest


class TestScripts(unittest.TestCase):
    scripts_path = "./scripts/"

    def run_script(self, script_name, **kwargs):
        script = f"python3 {script_name}.py"
        #  check if in the kwargs there is dirct key, for this one just add the value to the command
        if 'direct' in kwargs:
            script += " " + kwargs['direct']
            del kwargs['direct']
        params = ' '.join(f"{(len(k) > 1 and '--' or '-') + k} {v}" for k, v in kwargs.items())
        command = f"{script} {params}"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Log the output and errors
        print(f"\n{'-' * 60}\nRunning: {command}")
        print(f"Output: {output.decode()}")
        if error:
            print(f"Error: {error.decode()}")

        # Assert that return code is 0
        self.assertEqual(process.returncode, 0, f"{command} failed with return code {process.returncode}")

    def test_01_download_models(self):
        self.run_script("./download_models")

    def test_02_process_models(self):
        self.run_script("./process_models")

    def test_03_text_to_image(self):
        self.run_script(f"{self.scripts_path}text_to_image",
                        prompt="\"character, chibi, waifu, side scrolling, white background, centered\"",
                        num_images=1)

    #     python3 ./scripts/txt2img.py --num_images 2 --prompt 'A purple rainbow, filled with grass'

    def test_04_1_text_to_image(self):
        self.run_script(f"{self.scripts_path}txt2img", prompt="\"A purple rainbow, filled with grass\"", num_images=1)

    def test_05_embed_prompts(self):
        self.run_script(f"{self.scripts_path}embed_prompts",
                        prompts="\"A painting of a computer virus, An old photo of a computer scientist, A computer drawing a computer\"")

    # Add other test cases similarly...
    def test_06_generate_images_from_embeddings(self):
        self.run_script(f"{self.scripts_path}generate_images_from_embeddings", temperature=1.2,
                        ddim_eta=0.2)

    def test_07_generate_images_from_distributions(self):
        self.run_script(f"{self.scripts_path}generate_images_from_distributions", d=4, params_steps=1,
                        params_range='0.49 0.54',
                        num_seeds=4, temperature=1.2, ddim_eta=1.2)

    def test_08_generate_images_from_temperature_range(self):
        self.run_script(f"{self.scripts_path}generate_images_from_temperature_range", d=4, params_range='0.49 0.54',
                        params_steps=1,
                        temperature_steps=1, temperature_range='0.8 0.9')

    def test_09_generate_images_and_encodings(self):
        self.run_script(f"{self.scripts_path}generate_images_and_encodings",
                        prompt="\"An oil painting of a computer generated image of a geometric pattern\"",
                        num_iterations=1)

    def test_10_embed_prompts_and_generate_images(self):
        self.run_script(f"{self.scripts_path}embed_prompts_and_generate_images")

    def test_11_data_bounding_box_and_score_and_embedding_dataset(self):
        self.run_script(f"{self.scripts_path}data_bounding_box_and_score_and_embedding_dataset")

    def test_12_grid_generator(self):
        self.run_script("./utility/scripts/grid_generator", input_path="./test/test_images/clip_segmentation",
                        output_path="./tmp", rows=3, columns=2, img_size=256)

    def test_13_generate_images_from_random_prompt(self):
        self.run_script(f"{self.scripts_path}generate_images_from_random_prompt",
                        checkpoint_path="\"./input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors\"", cfg_scale=7, num_images=1,
                        output="\"./output/\"")

    def test_14_chad_sort(self):
        self.run_script(f"{self.scripts_path}chad_score",
                        direct="--model-path=\"input/model/chad_score/chad-score-v1.pth\" --image-path=\"./test/test_images/test_img.jpg\"")

    def test_15_chad_sort(self):
        self.run_script(f"{self.scripts_path}chad_sort",
                        direct="--dataset-path=\"test/test_zip_files/test-dataset-correct-format.zip\" --output-path=\"./output/chad_sort/\"")

    def test_16_run_generation_task(self):
        self.run_script(f"{self.scripts_path}run_generation_task",
                        task_path="\"./test/test_generation_task/generate_images_from_random_prompt_v1.json\"")

    def test_17_run_prompts_ga(self):
        self.run_script(f"{self.scripts_path}prompts_ga")


if __name__ == "__main__":
    unittest.main()
