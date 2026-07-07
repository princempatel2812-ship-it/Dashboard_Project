import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Local LLMs via Ollama
# Ensure you have run `ollama pull llama3` and `ollama pull mistral` locally
extractor_llm = Ollama(model="llama3", format="json") 
evaluator_llm = Ollama(model="mistral")

def process_pdf(file_path):
    print(f"Loading and chunking PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(pages)
    
    # Merging the first few highly relevant chunks for demonstration
    # In a full run, you would iterate through chapters or use RAG
    full_text = " ".join([chunk.page_content for chunk in chunks[:20]]) 
    return full_text

def extract_indicators(text):
    print("Extracting indicators using primary LLM (JSON Mode)...")
    prompt = f"""
    Analyze the following text from a UN Human Development Report.
    Extract the core numerical indicators, themes, and trends into this exact JSON structure:
    {{
        "country": "String",
        "HDI_value": Float,
        "HDI_rank": Integer,
        "life_expectancy_years": Float,
        "expected_years_of_schooling": Float,
        "mean_years_of_schooling": Float,
        "GNI_per_capita": Integer,
        "population": Integer,
        "themes": {{"economy": Integer, "education": Integer, "health": Integer, "inequality": Integer, "employment": Integer, "gender": Integer}},
        "key_strengths": ["String", "String"],
        "key_challenges": ["String", "String"],
        "hdi_trend": {{"1980": Float, "1990": Float, "2000": Float, "2010": Float, "2014": Float}}
    }}
    
    Source Text: {text[:6000]}
    """
    
    response = extractor_llm.invoke(prompt)
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        print("Failed to parse JSON. Raw output:", response)
        return None

def evaluate_extraction(source_text, extracted_json):
    print("Evaluating extraction quality using secondary LLM (LLM-as-a-Judge)...")
    prompt = f"""
    You are an AI evaluator checking the accuracy of a data extraction pipeline.
    
    Source Text snippet: {source_text[:2000]}
    
    Extracted JSON: {json.dumps(extracted_json)}
    
    Assess the quality based on Completeness, Consistency, and Factual Alignment. 
    Write a concise summary of the model's performance. Mention any trade-offs between accuracy and verbosity.
    """
    evaluation = evaluator_llm.invoke(prompt)
    return evaluation

if __name__ == "__main__":
    pdf_path = "zambiahumandevelopmentreport2016.pdf" 
    
    try:
        raw_text = process_pdf(pdf_path)
        extracted_data = extract_indicators(raw_text)
        
        if extracted_data:
            evaluation_report = evaluate_extraction(raw_text, extracted_data)
            
            output = {
                "data": extracted_data,
                "evaluation": evaluation_report
            }
            
            with open("extracted_data.json", "w") as f:
                json.dump(output, f, indent=4)
            print("Pipeline complete. Data saved to extracted_data.json")
    except Exception as e:
        print(f"Pipeline failed: {e}")