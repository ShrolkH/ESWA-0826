from langchain.prompts import PromptTemplate
from .self_llm import LLM_procket
from .prompt import (
    SUMMARY_PROMPT_SYSTEM,
    SUMMARY_PROMPT_USER,
    DOUBLE_SEARCH_USER_PROMPT,
    PROMPT_IMPROVE_COT
)

import json
import os
from datetime import datetime
from .esearch_pubmed import return_all_pmid
from .self_sqlite import ArticleDatabase
import re
import sqlite3


# Metadata extraction function for JSONLoader
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["year"] = record.get("pub_date").get('year')
    metadata["month"] = record.get("pub_date").get('month')
    metadata["day"] = record.get("pub_date").get('day')
    metadata["title"] = record.get("article_title")
    return metadata

# Function to create a prompt template
def create_prompt():
    prompt_template = """
    Your are a medical assistant for question-answering tasks. Answer the Question using the provided Context only. Your answer should be in your own words and be no longer than 128 words. \n\n Context: {context} \n\n Question: {question} \n\n Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def storage_input_and_output(graph_score,input, output, answer_type,config):
    # Get the current date, formatted as Year-Month-Day (e.g.: 2025-05-02)

    # Construct the file save path
    file_path = f'/hy-tmp/llm_lp-main/rag/llm_input_and_output/{config["current_date"]}.json'

    # Ensure the folder exists; create it if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Construct the data to be saved
    new_data = {
        "answer_type":answer_type,
        "input": input,
        "output": output,
        "pos": config['pos'],
        "graph_prediction":graph_score
    }

    # If the file exists, first read the existing data
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                # If the existing data is not of list type, convert it to a list
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Add the new data to the existing data
    existing_data.append(new_data)

    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


# Few-shot construction
def get_few_shot_data():
    few_data = []

    return few_data

# Use regular expressions to obtain revision comments
def get_thinking_text(text, pattern):
    # Use the re.DOTALL flag to make '.' match line breaks
    match = re.search(pattern, text, re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()  # Extract and strip leading and trailing whitespace
        print("Content to extract:")
        print(extracted_text)
    else:
        extracted_text = "Extraction error"
    print(f'Original text: {text}')
    print(f'Extraction result: {extracted_text}')
    return extracted_text

def extract_thinking_score(input_str):
    """
        Extract the floating-point number with two decimal places that follows the substring 'Scoring of the thinking:' in the string, and ensure the surrounding '**' symbols are also used as markers for matching.

        Parameters:
        input_str (str): Input string containing the target content

        Returns:
        float/None: The extracted floating-point number (returns None if not found or conversion fails)
    """
    # Regular Matching Pattern: Match the possible spaces following 'Scoring of the thinking:', then capture the number (including decimal points), and ensure the surrounding '**' symbols are also used as markers for matching
    pattern = r"\*\*Scoring of the thinking\*\*:\s*([\d]+\.[\d]{2})"
    match = re.search(pattern, input_str)
    
    if not match:
        print("No content related to 'Scoring of the thinking:' was found")
        return 0.00
    
    number_str = match.group(1)
    
    try:
        return float(number_str)
    except ValueError:
        print(f"Unable to convert '{number_str}' to a float")
        return None

# 读取JSON文件
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON format")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None
    
def get_reference_text_with_pmid(pmid_list):
    # File Path
    file_path = '/hy-tmp/llm_lp-main/data/combina_all_summary_time.json'

    # Read File Content
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result_data = []
    for item in data:
        if item['pmid'] in pmid_list:
            result_data.append(item['pmid'])
    
    print(f'Retrieved PMID: {result_data}')

    return result_data

def get_summary_by_llm(pmid_list,chroma):
    # Initialize the Database
    db = ArticleDatabase()
    llm = LLM_procket('qwen-plus-latest')
    references_text = ""
    num = 0
    for index, item in enumerate(pmid_list[:6]):
        # First check whether the abstract already exists in the database
        pmid = item.get("pmid", "")
        if "fallback" in pmid:
            continue
        else:
            # Obtain the abstract and pubdate
            summary_result = db.get_summary(pmid) if pmid else (None, None)
            existing_summary, existing_pubdate = summary_result if summary_result else (None, None)
            
            if existing_summary:
                # If the abstract already exists in the database, use it directly
                reply = existing_summary
                
                # Check if pubdate is empty; if it is empty and there is a pubdate in the item, update the database
                if not existing_pubdate and 'pubdate' in item and item['pubdate']:
                    db.update_pubdate(pmid, item['pubdate'])
            else:
                # If the abstract is not in the database, generate a new abstract
                user_prompt = SUMMARY_PROMPT_USER.format(
                    title=item['article_title'],
                    abstract=item['article_abstract']
                )
                message = [
                    {
                        "role": "system", "content": SUMMARY_PROMPT_SYSTEM
                    },
                    {
                        "role": "user", "content": user_prompt
                    }
                ]
                
                # Generate the Abstract
                reply = llm.chat_with_llm(message)
                
                # Save to the database, including pubdate
                if pmid:
                    pubdate = item.get('pubdate', None)
                    db.save_article(pmid, item['article_title'], item['article_abstract'], reply, pubdate)
                    # Add to the vector database
                    similary_text = chroma.similarity_search(item['article_abstract'],k=4)
                    doc,score = similary_text[0]
                    if score > 0.1: # The smaller the score, the higher the similarity
                        chroma.create_vectordb_from_sqlite_db(
                            [
                                {
                                    "pmid":pmid,
                                    "pub_date":pubdate,
                                    "title":item['article_title'],
                                    "article_abstract":reply,
                                    "pub_date":pubdate
                                }
                            ]
                        )
            num += 1    
            references_text += f"{num}. {reply}"
    return references_text



# Function to perform RAG-based chat
def rag_chat(config,chroma,query,graph_score):
    question ="\n## Question: \n" + f'Is there a potential association between the diseases "{query["src_name"]}" and "{query["dst_name"]}"? '
    if graph_score == 1:
        graph_prediction = "## Structural prediction is :\n Predictions based on four indicators, namely, Jaccard's similarity coefficient, resource allocation index, prioritized connectivity, and Adham-Adhar index, resulted in a correlation between the two."
    else:
        graph_prediction = "## Structural prediction is :\n Predictions based on four indicators, namely, Jaccard similarity coefficient, resource allocation index, prioritized connectivity, and Adham-Adhar index, resulted in no correlation between the two."
    self_llm = LLM_procket(config['model_name'],temperature=0)
    if config['is_RAG']:
        references_list = return_all_pmid(query['src_name'],query['dst_name'],last_time=config['last_time'])
        # Obtain explanations for disease comorbidity or single diseases
        new_summary_text = get_summary_by_llm(references_list,chroma)

        # If the explanations for comorbidity or single diseases are empty, perform semantic retrieval
        if new_summary_text == "":
            num_id = 0
            search_text = f'{query["src_name"]},is defined as {query["src_dec"]}, {query["dst_name"]},is defined as {query["dst_dec"]}'
            references_summary_list = chroma.similarity_search(search_text,timestamp_threshold=config['timestamp_threshold'],k=4)
            references_text = ""
            for index, (doc, score) in enumerate(references_summary_list):
                if score < 0.6: # ChatGPT holds that: ≤ 0.2 – strongly relevant; ≤ 0.4 – moderately relevant; ≤ 0.5 – weakly relevant (acceptable)
                    num_id += 1
                    references_text += f"{num_id}. {doc.page_content}\n"
            print("Semantic retrieval has been executed")
            new_summary_text = references_text

        references_text = '\n## Abstract (Medical literature abstracts relevant to the question): \n' + new_summary_text
        node_info = f"""## Node information is:
In the current disease-symptom network, the node "{query["src_name"]}" is defined as "{query["src_dec"]}". It is connected to {query["src_degree"]} nodes. The node "{query["dst_name"]}" is defined as "{query["dst_dec"]}", and it is connected to {query["dst_degree"]} nodes. They share {query["common_count"]} common nodes: {query["common_neighbors_names_text"]}\n
"""
        node_info += graph_prediction
        query['question'] = question
        query['context'] = references_text
        query['node_info'] = node_info

        input_text =  DOUBLE_SEARCH_USER_PROMPT.format(
            graph_info_text=query['graph_info_text'],
            context=references_text,
            question=question,
            node_info=node_info
        )
        rag_message = [
            {"role": "system",   "content":     PROMPT_IMPROVE_COT},
        ]
        few_data = get_few_shot_data()
        for item in few_data:
            rag_message.append(item)

        rag_message.append({"role": "user",   "content": input_text})
        llm_response = self_llm.chat_with_llm(rag_message)
        message = rag_message
        storage_input_and_output(graph_score,rag_message,llm_response,answer_type="answer",config=config)
        return llm_response




