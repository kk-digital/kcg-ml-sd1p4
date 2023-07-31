import unittest
import subprocess


class TestScripts(unittest.TestCase):
    scripts_path = "./scripts/"

    def run_script(self, script_name, **kwargs):
        script = f"python3 {self.scripts_path}{script_name}.py"
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

    def test_1_process_base_model(self):
        self.run_script("process_base_model", direct="'input/model/v1-5-pruned-emaonly.safetensors'")

    def test_2_text_to_image(self):
        self.run_script("text_to_image", prompt="'character, chibi, waifu, side scrolling, white background, centered'",
                        checkpoint_path="'./input/model/v1-5-pruned-emaonly.safetensors'", batch_size=1, num_images=1)

    def test_3_embed_prompts(self):
        self.run_script("embed_prompts",
                        prompts="'A painting of a computer virus, An old photo of a computer scientist, A computer drawing a computer'")

    # Add other test cases similarly...
    def test_4_generate_images_from_embeddings(self):
        self.run_script("generate_images_from_embeddings", num_seeds=4, temperature=1.2, ddim_eta=0.2)

    def test_5_generate_images_from_distributions(self):
        self.run_script("generate_images_from_distributions", d=4, params_steps=4, params_range='0.49 0.54',
                        num_seeds=4, temperature=1.2, ddim_eta=1.2)

    def test_6_generate_images_from_temperature_range(self):
        self.run_script("generate_images_from_temperature_range", d=4, params_range='0.49 0.54', params_steps=3,
                        temperature_steps=3, temperature_range='0.8 2.0')

    def test_7_generate_images_and_encodings(self):
        self.run_script("generate_images_and_encodings",
                        prompt="'An oil painting of a computer generated image of a geometric pattern'",
                        num_iterations=10)

    def test_8_embed_prompts_and_generate_images(self):
        self.run_script("embed_prompts_and_generate_images")

    def test_9_data_bounding_box_and_score_and_embedding_dataset(self):
        self.run_script("data_bounding_box_and_score_and_embedding_dataset")


if __name__ == "__main__":
    unittest.main()
