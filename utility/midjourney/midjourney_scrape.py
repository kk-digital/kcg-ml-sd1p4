import json
import csv
import tiktoken

# source midjourney json
data_json = "./input/prompts (2).json"
# full path of output csv file
output_csv = "./input/midjourney_phrases_database.csv"
# initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

f = open(data_json)
json_content = json.load(f)

phrase_id = {}
phrase_total_count = {}
phrase_positive_count = {}
phrase_negative_count = {}

i = 0
x = 0
for item in json_content:
   

    if "prompt" in item:
        positive_prompt = item["prompt"]
        i+=1
      

    if  len(item["negative_prompt"]) != 0:
        negative_prompt = item["negative_prompt"]
       
        negative_prompt = negative_prompt.split('--no')
        negative_prompt = list(filter(None, negative_prompt))

        negative_prompt = ','.join(negative_prompt)
        
        x+=1

        negative_prompt = negative_prompt.replace('\n', ',')
        negative_prompt = negative_prompt.replace('\"', '')
        negative_prompt = negative_prompt.replace(',,', ',')
        negative_prompt = negative_prompt.replace(' , ', ',')
        negative_prompt = negative_prompt.replace(' ,', ',')
        negative_prompt = negative_prompt.replace(', ', ',')
        negative_prompt = negative_prompt.replace(' . ', ',')
        negative_prompt = negative_prompt.replace(' .', ',')
        negative_prompt = negative_prompt.replace('. ', ',')
        negative_phrases = negative_prompt.split(',')

        # negative
        for phrase in negative_phrases:
            # lowercase
            phrase = phrase.lower()
            # remove leading and trailing space
            phrase = phrase.strip()

          
            if phrase not in phrase_id:
                phrase_id[phrase] = len(phrase_id)

            if phrase in phrase_total_count:
                count = phrase_total_count[phrase]
                count += 1
                phrase_total_count[phrase] = count
            else:
                phrase_total_count[phrase] = 1

            if phrase in phrase_negative_count:
                count = phrase_negative_count[phrase]
                count += 1
                phrase_negative_count[phrase] = count
            else:
                phrase_negative_count[phrase] = 1


    positive_prompt = positive_prompt.replace('\n', ',')
    positive_prompt = positive_prompt.replace('\"', '')
    positive_prompt = positive_prompt.replace(',,', ',')
    positive_prompt = positive_prompt.replace(' , ', ',')
    positive_prompt = positive_prompt.replace(' ,', ',')
    positive_prompt = positive_prompt.replace(', ', ',')
    positive_prompt = positive_prompt.replace(' . ', ',')
    positive_prompt = positive_prompt.replace(' .', ',')
    positive_prompt = positive_prompt.replace('. ', ',')
    positive_phrases = positive_prompt.split(',')
    positive_phrases = list(filter(None, positive_phrases))

    

    # positive
    for phrase in positive_phrases:
        # lowercase
        phrase = phrase.lower()
        # remove leading and trailing space
        phrase = phrase.strip()
       
        if phrase not in phrase_id:
            phrase_id[phrase] = len(phrase_id)

        if phrase in phrase_total_count:
            count = phrase_total_count[phrase]
            count += 1
            phrase_total_count[phrase] = count
        else:
            phrase_total_count[phrase] = 1

        if phrase in phrase_positive_count:
            count = phrase_positive_count[phrase]
            count += 1
            phrase_positive_count[phrase] = count
        else:
            phrase_positive_count[phrase] = 1

    


print("Total positive prompts: {}".format(i))
print("Total neg prompts: {}".format(x))
print("Total phrases: {}".format(len(phrase_id)))

# sort
sorted_dict = {}
sorted_keys = sorted(phrase_total_count, key=phrase_total_count.get, reverse=True)
for w in sorted_keys:
    sorted_dict[w] = phrase_total_count[w]

# save csv
# field names
fields = ['index', 'total count', 'positive count', 'negative count', 'token size', 'phrase str']

# writing to csv file
with open(output_csv, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    for key, val in sorted_dict.items():
        if key == "":
            continue

        # compute token size
        phrase_tokens = enc.encode(key, allowed_special={'<|endoftext|>'})
        num_tokens = len(phrase_tokens)
        positive_count = phrase_positive_count[key] if key in phrase_positive_count else 0
        negative_count = phrase_negative_count[key] if key in phrase_negative_count else 0
        row = [phrase_id[key], val, positive_count, negative_count, num_tokens, key]
        csvwriter.writerow(row)

