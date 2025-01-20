import requests
import time

from pydantic import BaseModel, Field
from typing import Optional, Iterator, Callable

import streamlit as st
from pypdf import PdfReader
from markdown_pdf import MarkdownPdf, Section

from phi.workflow import Workflow, RunResponse, RunEvent

from phi.tools.jina_tools import JinaReaderTools
from phi.agent import Agent
from phi.model.google import Gemini
# from phi.model.groq import Groq
# from phi.model.huggingface import HuggingFaceChat
# from phi.model.deepseek import DeepSeekChat
# from phi.model.openrouter import OpenRouter
# from phi.model.mistral import MistralChat
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.utils.log import logger
from phi.utils.pprint import pprint_run_response

from dotenv import load_dotenv
import os
from jobspy import scrape_jobs
import json

load_dotenv()
# Model definitions (Gemini and Groq are used in this example)
MODEL_GEMINI: Gemini = Gemini(id="gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"))
# MODEL_GROQ: Groq = Groq(id="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
JINA_API_KEY=os.getenv("JINA_API_KEY")

class LoggerManager:
    def __init__(self, logger, main_status_placeholder=None):
        self.logger = logger
        self.main_status_placeholder = main_status_placeholder

    def log(self, message):
        self.logger.info(message)
        if self.main_status_placeholder:
            self.main_status_placeholder.write(message)


class JobOffer(BaseModel):
    title: str = Field(..., description="Title of the Job Offer.")
    company: str = Field(..., description="Company.")
    description: Optional[str] = Field(..., description="Description of the Job Offer.")

class JobOffers(BaseModel):
    offers: list[JobOffer] = Field(..., description="List of Job Offers.")

class Keyword(BaseModel):
    keyword: str = Field(..., description="Keyword, it can be either single word or an expression")

class KeywordFrequency(BaseModel):
    keyword: str = Field(..., description="Keyword, it can be either single word or an expression")
    frequency: int = Field(..., description="How many job offers listed this keyword")

class ListKeywordsFrequency(BaseModel):
    keywords: list[KeywordFrequency] = Field(..., description="List of Keywords.")

class ListKeywords(BaseModel):
    keywords: list[str] = Field(..., description="List of Keywords. It can contains repated keywords.")

class SpecificJobOffer(BaseModel):
    company: str = Field(..., description="Company name.")
    title: str = Field(..., description="Title of the Job Offer.")
    description: str = Field(..., description="Content of the Job Offer.")
    keywords: list[str] = Field(..., description="List of Keywords. It can contains repated keywords.")


def get_latest_jobs_for_profession(profession: str, qty: int = 1):
    """
    This function searches for the latest job offers for a specific profession.

    Args:
        profession (str): Desired profession for job openings.
        location (str): Desired location for jobs. Defaults to empty string.

    Returns:
        list: List of dictionaries containing job offer information.
    """

    # Defines the sites to be searched
    sites = ["indeed", "linkedin", "glassdoor", "google"]

    jobs = scrape_jobs(
        site_name=sites,
        search_term=profession,
        location="",
        results_wanted=qty, 
        hours_old=168
    )

    if jobs is not None and not jobs.empty:
        jobs = jobs[["company", "title", "description"]]
        jobs = jobs.fillna("")
        jobs_list = jobs.to_dict(orient="records")  # Converts to a list of dictionaries
        # logger_manager.log(f"Jobs list: {jobs_list}")  # Log for debugging

        # Creates a list of JobOffer objects
        job_offers = []
        for job_dict in jobs_list:
            try:
                # Ensures that all required fields are present
                if all(key in job_dict for key in ["company", "title", "description"]):
                    job_offers.append(JobOffer(
                        title=str(job_dict["title"]),
                        company=str(job_dict["company"]),
                        description=str(job_dict["description"])
                    ))
            except Exception as e:
                logger.error(f"Error creating JobOffer: {e}")
                continue

        if job_offers:
            return JobOffers(offers=job_offers).model_dump_json()

    logger.warning("No valid job offers found")
    return JobOffers(offers=[])

class RecommendationGenerator(Workflow):
    extractor_job_offer: Agent = Agent(model=MODEL_GEMINI,
            description=f"You are a famous leadership expert in HR Analysts. You excel at extracting titles, skills, keywords from job offers.",
            instructions=[f"Given a job offer, extract company name, title and as many keywords as possible related specifically to the profession. ",
                          "Search for skills (human related or technical), methodologies and technologies (including tools), etc.",
                          "Ensure the keywords are related to the description of the role itself and not about the company or about the benefits. ",
                          "The result should be the content of the job offer and a list of keywords in the response model format."],
            response_model=SpecificJobOffer, markdown=True)
    

    extractor: Agent = Agent(model=MODEL_GEMINI,
            description=f"You are a famous leadership expert in HR Analysts. You excel at extracting titles, skills, keywords from job offers.",
            instructions=[f"Given a list of job offers, extract as many keywords as possible related specifically to the profession. ",
                          "Search for profession titles, jargons of the profession, skills (human related or technical), methodologies and technologies (including tools), etc."
                          "Iterate over each job offer and add their extracted keywords in the final list. That is, the list could contain repeated keywords from different job offers (not from the same job offer), ignoring the job offer they were extracted from."
                          "Ensure the keywords are related to the role itself and not about the company or about the benefits. ",
                          "The result should be a list of keywords in the response model format."],
            response_model=ListKeywords, markdown=True)

    prioritizer: Agent = Agent(model=MODEL_GEMINI,
            description=f"You excel at evaluating keywords and the frequency with which they occur. You are accurate and do not fabricate data.",
            instructions=[f"Evaluate the list of keywords and tell me the frequency of each one",
                          "Sort the keywords by the frequency in descending order",
                          "The result should be in the response model format."],
            response_model=ListKeywordsFrequency, markdown=True)

    recommender: Agent = Agent(model=MODEL_GEMINI,
                description="You are a famous HR analyst. You are very good at suggesting improvements to resumes based on a list of keywords and best practices for resume writing.",
                instructions=["The resume points you should evaluate: Headline, Summary, Work Experience (experience title and experience summary)",
                    "Be specific about which section above you will make a suggestion for, show how it currently is, and how you suggest changing it based on the given keywords and relate it with the context of the person's experience.",
                    "If specific_job_offer_content and specific_job_offer_keywords are provided, prioritize adapting the whole resume to that job offer and keywords, but also you should consider the all the other keywords provided.",
                    "Review the details of each company mentioned in the person's experience, evaluating all keywords. Give a final suggestion for each title and summary experience, reasoning and relating the keywords",
                    "Each suggestion of keywords should be contextual and only suggest if infered that the person has this experience, skill or knowledge, otherwise create a separate suggestion confirming if the person has this experience",
                    "When evaluating the title and summary of the resume take into account the whole experience of the person in each company, so you can suggest most contextual keywords and justify the link with specific experiences.",
                    "Always use this topics evaluation: Current Info, Detailed Reasoning (mentioning keywords), Suggested Text - Alternative 1, Suggested Text - Alternative 2",
                    "You don't need to explain the evaluation structure to the user.",
                    "Your result should always be in the same language of the resume"], 
                    expected_output="The result should be in a beautiful structured content in Markdown but without the marks ```markdown.",
                    markdown=True)

    final_recommender: Agent = Agent(model=MODEL_GEMINI,
                description="You are a famous HR analyst. You are very good at suggesting improvements to resumes based on a list of keywords and best practices for resume writing.",
                instructions=["Given the recommended alternatives, integrate the best nuances and give a final improved resume so the person could share it as it is.",
                    "Your result should always be in the same language of the resume"], 
                    expected_output="The result should be in a beautiful structured resume in Markdown but without the marks ```markdown.",
                    markdown=True)

    def run(self, profession: str, curriculum_vitae_file_name: str, curriculum_vitae_content: str, link_job_offer: str = None, qty: int = 1, use_cache: bool = True) -> RunResponse:
        logger_manager.log(f"🚀 Getting the last job offers and generating recommendations: {profession}")
        sorted_keywords = None

        if use_cache and "cached_profession" in self.session_state:
            logger_manager.log("🗄️ Checking if keywords from job offers exists in cache")
            for cached_profession in self.session_state["cached_profession"]:
                if "profession" in cached_profession and cached_profession["profession"] == profession:
                    logger_manager.log("🗄️ Found cached keywords")
                    sorted_keywords = cached_profession["sorted_keywords"]

        search_results: Optional[JobOffers] = None
        if not sorted_keywords or sorted_keywords.get('keywords', []) == []:
            logger_manager.log("🗄️  No cached keywords found.")
            logger_manager.log("🚀  Search the job offers related to profession.")
            # Step 1: Search the job offers related to profession
            num_tries = 0
            # Run until we get a valid job offers
            while search_results is None and num_tries < 3:
                try:
                    num_tries += 1

                    logger_manager.log(f"🚀 Searching job offers for {profession}")
                    jobs = get_latest_jobs_for_profession(profession, qty=qty)
                    searcher_response = RunResponse(content=jobs)

                    if searcher_response and searcher_response.content:
                        search_results = JobOffers.model_validate_json(searcher_response.content)
                        logger.warning("🔍 Found {search_results.offers}...")
                    else:
                        search_results = None
                        logger.warning("🔍 Searcher response invalid, trying again...")
                except Exception as e:
                    logger.warning(f"Error running searcher: {e}")

            # If no search_results are found for the topic, end the workflow
            if search_results is None or len(search_results.offers) == 0:
                return RunResponse(
                    run_id=self.run_id,
                    event=RunEvent.workflow_completed,
                    content={"recommendation": f"Sorry, could not find any job offer related to profession: {profession}", "keywords_frequency": [], "specific_job_offer": None}                    
                )

            # Step 2: Extract keywords in batches
            logger_manager.log("🚀 Extracting keywords from job offers")

            job_offers = [v.model_dump() for v in search_results.offers]
            all_keywords = [] # Initialize an empty list to accumulate keywords

            for i in range(0, len(job_offers), 3):
                batch_offers = job_offers[i:i+3]
                extractor_input = {
                    "profession": profession,
                    "offers": batch_offers,
                }

                logger_manager.log(f"🚀 Extracting keywords from job offers batch {i // 3 + 1}")
                keywords_response: RunResponse = self.extractor.run(json.dumps(extractor_input, indent=4))

                # Accumulate keywords
                if keywords_response.content and keywords_response.content.keywords:
                    all_keywords.extend(keywords_response.content.keywords)

            # Step 3: Sorting the keywords
            logger_manager.log("🗂️ Sorting the keywords")
            prioritizer_input = {
                "keywords": all_keywords
            }
            sorted_keywords_response: RunResponse = self.prioritizer.run(json.dumps(prioritizer_input, indent=4))
            logger_manager.log("✅ Sorting the keywords...DONE")
            # logger_manager.log(f"✅ Sorted:...{sorted_keywords_response.content}")

            if sorted_keywords_response.content and sorted_keywords_response.content.keywords:
                sorted_keywords = sorted_keywords_response.content
            else:
                sorted_keywords = ListKeywordsFrequency(keywords=[])

            if "cached_profession" not in self.session_state:
                self.session_state["cached_profession"] = []
            self.session_state["cached_profession"].append({"profession": profession, "sorted_keywords": sorted_keywords.model_dump()})
            logger_manager.log(f"🗄️ Saved in the cache")

        # Step 4: Extract keywords from a specific link
        headers = {
            'Authorization': f'Bearer {JINA_API_KEY}'
        }

        url = f'https://r.jina.ai/{link_job_offer}'
        response = requests.get(url, headers=headers)
        job_offer_content = response.text

        specific_job_offer = None
        if link_job_offer:
            logger_manager.log(f"🚀 Extracting keywords from specific job offer in {link_job_offer}")
            extractor_input = {
                    "profession": profession,
                    "offer": f"{job_offer_content}",
            }
            specific_job_offer_response: RunResponse = self.extractor_job_offer.run(json.dumps(extractor_input, indent=4))
            if specific_job_offer_response.content and specific_job_offer_response.content.keywords:
                specific_job_offer = specific_job_offer_response.content
        
        # Step 5: Exploring improvements
        logger_manager.log("✍️ Generating recommendation")
        keywords = sorted_keywords.get('keywords', []) if isinstance(sorted_keywords, dict) else [v.model_dump() for v in sorted_keywords.keywords]
        recommender_input = {
            "curriculum_vitae_content": curriculum_vitae_content,
            "specific_job_offer_content": specific_job_offer.description if specific_job_offer else "",
            "specific_job_offer_keywords": specific_job_offer.keywords if specific_job_offer else [],
            "profession": profession,
            "keywords_frequency": keywords
        }
        recommendation: RunResponse = self.recommender.run(json.dumps(recommender_input, indent=4), stream=False, markdown=True)
        
        # Step 6: Generating Improved Resume
        
        logger_manager.log("✍️ Generating improved resume")
        recommender_input = {
            "recommendation": recommendation.content
        }
        improved_resume: RunResponse = self.improved_resume.run(json.dumps(recommender_input, indent=4), stream=False, markdown=True)
        
        logger_manager.log("✅ Generating recommendation: DONE")
        # return recommendation
        result = RunResponse(content={"recommendation": recommendation, "improved_resume": improved_resume, "keywords_frequency": keywords, "specific_job_offer": specific_job_offer})
        return result

st.title("Resume Evaluator")

# Sidebar for inputs
with st.sidebar:
    user_api_key = None
    api_key_option = st.selectbox("🤖 AI LLM Gemini API Key:", ("Use Default Key", "Enter Custom Key"))
    if api_key_option == "Enter Custom Key":
        user_api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if not user_api_key:
            st.error("You need to enter a Gemini API key.")
            st.stop()
    elif api_key_option == "Use Default Key":
        if not os.getenv("GEMINI_API_KEY"):
            st.error("The Gemini API key is not set. Enter a custom key or configure the .env file.")
            st.stop()

    api_key = user_api_key or os.getenv("GEMINI_API_KEY")

    MODEL_GEMINI: Gemini = Gemini(id="gemini-2.0-flash-exp", api_key=api_key)

    profession = st.text_input("For which profession do you want to refine your resume?", key="profession_input")
    qty = st.number_input("Max vacancies to search in each platform", min_value=1, max_value=10, value=5, step=1, label_visibility="visible")
    
    curriculo_pdf = st.file_uploader("Upload your resume in PDF format", type=["pdf"], key="curriculo_pdf_input")
    curriculo_file_name = ""
    curriculo_conteudo = ""

    pdf_status_placeholder = st.empty()

    link_job_offer = st.text_input(
        "Job Posting URL (optional)", key="link_job_offer_input",
        placeholder="Paste the link here",
        help="Optional. If filled, the resume suggestions will be personalized for this job. Otherwise, your area of expertise and recent job postings in the area will be considered.",
    )
    

    # Checkbox to force new job search
    search_new_jobs = st.checkbox("Use cached keywords", value=True, 
        help="If checked, use the cached keywords from job board vacancies instead of searching again. If you're conducting an evaluation for this profession for the first time, a new search will be carried out.",)
    
    # Check if both required fields are filled
    is_button_disabled = not (profession and curriculo_pdf)
    # if not profession:
    #     st.warning("Please enter a profession")
    # if not curriculo_pdf:
    #     st.warning("Please upload a PDF resume")

    if st.button("Evaluate resume", disabled=is_button_disabled, use_container_width=True):
        st.session_state['run_evaluation'] = True
    else:
        st.session_state['run_evaluation'] = False


# Main content area
with st.expander("How does it work?", expanded=(not st.session_state.get('run_evaluation', False))):
    st.markdown("""
    ### How does it work?

    Before you begin, here's a summary of how the system works to ensure you know what to expect:
    """)

    st.markdown(f"""
    1. **Job Portal Search:** The system searches for vacancies on Indeed, LinkedIn, Glassdoor, and Google Jobs using the provided profession to find contextual keywords.
    2. **Keyword Extraction:** The vacancies found are analyzed to extract relevant keywords for the profession.
    3. **Keyword Prioritization:** The extracted keywords are evaluated and prioritized based on the frequency of occurrence.
    4. Specific Job Link Analysis (Conditional): If a specific job posting link is provided by the user, the system analyzes that specific job description to extract highly relevant keywords tailored to that particular role. 
    5. **Recommendation Generation:** Based on the keywords, the system generates specific recommendations to improve the resume.
    """)

# Status and results area
sidebar_status_placeholder = st.empty()
main_status_placeholder = st.empty()
logger_manager = LoggerManager(logger, main_status_placeholder)

if curriculo_pdf:
    curriculo_file_name = curriculo_pdf.name
    with st.sidebar:
        pdf_status_placeholder.write("PDF resume uploaded successfully!")
        pdf_status_placeholder.write("Converting PDF to text...")

    try:
        pdf_reader = PdfReader(curriculo_pdf)
        for page in pdf_reader.pages:
            curriculo_conteudo += page.extract_text() + "\n\n"
        with st.sidebar:
            pdf_status_placeholder.write("PDF converted to text successfully!")
    except Exception as e:
        with st.sidebar:
            pdf_status_placeholder.write("Error processing the PDF: {e}")
        st.stop()

if st.session_state.get('run_evaluation', False):
    with st.spinner('Running...'):
        # Create the workflow
        generate_recommendation = RecommendationGenerator(
            session_id=f"recommendation-to-{profession}",
            storage=SqlWorkflowStorage(
                table_name="recommendation_workflows",
                db_file="tmp/workflows.db",
            ),
        )

        main_status_placeholder.empty()

        # Run workflow
        agent_recommendation = generate_recommendation.run(
            profession=profession,
            curriculum_vitae_file_name=curriculo_file_name,
            curriculum_vitae_content=curriculo_conteudo,
            link_job_offer=link_job_offer,
            qty=qty,
            use_cache=not search_new_jobs
        )

        main_status_placeholder.empty() 

        with st.expander("Keywords from several job offers", expanded=False):
            st.markdown(agent_recommendation.content["keywords_frequency"])

        if agent_recommendation.content["specific_job_offer"]:
            company = agent_recommendation.content["specific_job_offer"].company
            title = agent_recommendation.content["specific_job_offer"].title
            keywords = agent_recommendation.content["specific_job_offer"].keywords
            with st.expander(f"Keywords from the {company}: {title}", expanded=False):
                st.markdown(keywords)

        markdown_recommendation_text=agent_recommendation.content["recommendation"].content
        markdown_resume_text=agent_recommendation.content["improved_resume"].content

        with st.expander("Copy the raw result", expanded=False):
            st.markdown(f"""
                    ````markdown
                    {markdown_resume_text}
            """)

        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section(markdown_resume_text), paper_size="A4-L")
        pdf.meta["title"] = f"Resume - {profession}"
        
        timestamp = str(int(time.time()))
        pdf_file_name = f"resume_{profession}_{timestamp}.pdf"
        pdf.save(pdf_file_name)

        with open(pdf_file_name, 'rb') as f:
           st.download_button('Download the new resume', f, file_name=pdf_file_name) 

        st.markdown(markdown_resume_text)
