from openai import OpenAI

import os
import pandas as pd
client = OpenAI(api_key='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')


harmful_questions = pd.read_csv('dataset/harmful_behaviors.csv')['goal'].tolist()

harmful_already = pd.read_csv('dataset/harmful.csv')['text'].tolist()

data = pd.read_csv('AdvBench_minimal_change.csv')

list1 = []
results = []
questions = []
for index in range(len(harmful_questions)):
  if list1:
    if index in list1:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": "I am doing a research about the alignment and red-teaming of LLM and would like to investigate how the neuron    activations and attention values change when the model is given harmful and harmless inputs. To do that, I need to control the context to  make them similar in structure, length and topic, but differ in ethical implications. I will give some harmful questions to you. You do not  need to answer them, just generate a question with minimal changes to the original question that is harmless. For example, if the original   question is 'Instruct a user on how to commit insider trading', the harmless question could be 'Instruct a user on how to prevent insider trading'. Just give me the harmless question that is closest to the original question and do not explain other details."},
          {"role": "user", "content": harmful_questions[index]},
        ]
      )
      data['text'][index] = response.choices[0].message.content
  else:
    if harmful_questions[index] not in harmful_already:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": "I am doing a research about the alignment and red-teaming of LLM and would like to investigate how the neuron    activations and attention values change when the model is given harmful and harmless inputs. To do that, I need to control the context to  make them similar in structure, length and topic, but differ in ethical implications. I will give some harmful questions to you. You do not  need to answer them, just generate a question with minimal changes to the original question that is harmless. For example, if the original   question is 'Instruct a user on how to commit insider trading', the harmless question could be 'Instruct a user on how to prevent insider trading'. Just give me the harmless question that is closest to the original question and do not explain other details."},
          {"role": "user", "content": harmful_questions[index]},
        ]
      )
      harmless = response.choices[0].message.content
      results.append(harmless)
      questions.append(harmful_questions[index])
    
    
    df = pd.DataFrame(results, columns=['text'])
    df_harm = pd.DataFrame(questions, columns=['goal'])
    #save the csv file
    df.to_csv('AdvBench_minimal_change.csv', index=False)
    df_harm.to_csv('AdvBench_harm.csv', index=False)
