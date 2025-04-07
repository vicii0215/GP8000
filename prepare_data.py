import pandas as pd
import json

def prompt1():
    def row_to_prompt(row):
        try:
            age = str(float(row['Age']))
            gender = str(row['Gender']).strip().lower()
            bedtime = str(row['Bedtime'])
            wakeup = str(row['Wakeup time'])
            duration = str(float(row['Sleep duration']))
            rem = str(float(row['REM sleep percentage']))
            deep = str(float(row['Deep sleep percentage']))
            light = str(float(row['Light sleep percentage']))
            awakenings = str(float(row['Awakenings']))
            caffeine = str(row['Caffeine consumption'])
            alcohol = str(row['Alcohol consumption']).lower()
            smoking = str(row['Smoking status']).lower()
            exercise = str(row['Exercise frequency'])

            prompt = (
                f"A {age}-year-old {gender} goes to bed at {bedtime} and wakes up at {wakeup}, "
                f"sleeping for {duration} hours. "
                f"{'His' if gender == 'male' else 'Her'} REM sleep accounts for {rem}%, deep sleep {deep}%, "
                f"and light sleep {light}% of the total duration. "
                f"{'He' if gender == 'male' else 'She'} wakes up {awakenings} times during the night. "
                f"{'He' if gender == 'male' else 'She'} drinks {caffeine} of coffee daily, consumes alcohol {alcohol}, "
                f"{'does not smoke' if smoking == 'non-smoker' else 'smokes'}, and exercises {exercise}. "
                f" What is {'his' if gender == 'male' else 'her'} predicted sleep efficiency? Directly give a value between 0 and 1 and don't give the process of throught."
            )
            return prompt
        
        except Exception as e:
            print(f"Error processing row: {e}")
            return ""

    def generate_prompts(input_csv, output_csv, results_csv):
        df = pd.read_csv(input_csv)

        # personal information
        df['Prompt'] = df.apply(row_to_prompt, axis=1)
        df['Prompt'].to_csv(output_csv, index=False)

        # value of sleep efficiency
        df_labels = df[['ID', 'Sleep efficiency']]
        df_labels.to_csv(results_csv, index=False)

        print(f"Prompts saved to {output_csv}, Truth label saved to {results_csv}")


    generate_prompts("raw_dataset/Sleep_Efficiency.csv", "prompts/prompts_QA.csv", "truth.csv")


def prompt2():
    def row_to_prompt_json(row):
        try:
            data = {
                "Age": float(row["Age"]),
                "Gender": str(row["Gender"]).strip(),
                "Bedtime": str(row["Bedtime"]).strip(),
                "Wakeup time": str(row["Wakeup time"]).strip(),
                "Sleep duration": float(row["Sleep duration"]),
                "REM sleep percentage": float(row["REM sleep percentage"]),
                "Deep sleep percentage": float(row["Deep sleep percentage"]),
                "Light sleep percentage": float(row["Light sleep percentage"]),
                "Awakenings": float(row["Awakenings"]),
                "Caffeine consumption": str(row["Caffeine consumption"]).strip(),
                "Alcohol consumption": str(row["Alcohol consumption"]).strip(),
                "Smoking status": str(row["Smoking status"]).strip(),
                "Exercise frequency": str(row["Exercise frequency"]).strip()
            }

            json_str = json.dumps(data, indent=2)
            prompt = f"Input:\n{json_str}\n\n Task: What is the sleep efficiency for him/her?. Directly output a constant between 0 and 1"
            return {"prompt": prompt}
        
        except Exception as e:
            print(f"Error processing row: {e}")
            return {"prompt": ""}

    def generate_json_prompt_file(input_csv, output_json):
        df = pd.read_csv(input_csv)
        prompts = df.apply(row_to_prompt_json, axis=1).tolist()

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        print(f"JSON prompts saved to {output_json}")

    # 示例调用
    generate_json_prompt_file("raw_dataset/Sleep_Efficiency.csv", "prompts/prompts_json.json")

prompt1()
prompt2()