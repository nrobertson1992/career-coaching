# For web scraping transcription URLs.
import requests
import json

# For getting transcription URLs fron edX's db.
import snowflake.connector

# For loading to user data schema.
from snowflake.connector.pandas_tools import write_pandas

# For data transformation.
import pandas as pd
import numpy as np

# For printing time it takes to translate.
from datetime import datetime
import time

# Multiprocessing
from multiprocessing.pool import ThreadPool as Pool

# Progress tracking.
from tqdm import tqdm


# For generating different keys for components.
import string
import random

# For unique streamlit bits.
import streamlit as st
from annotated_text import annotated_text
from streamlit_pills import pills

# Set up open ai-chatgpt.
import openai
openai.api_key = st.secrets['info']['open_ai_api_key']

# Initiate Snowflake client. Be sure to delete your password before you share notebooks.
ctx = snowflake.connector.connect(
    user=st.secrets['info']['user'],
    password=st.secrets['info']['password'],
    account=st.secrets['info']['account'],
    warehouse=st.secrets['info']['warehouse'],
    database=st.secrets['info']['database'],
    role=st.secrets['info']['role'],
)

## Function for extracting data from database into tabular format.
def run_query(query, columns,replace=False):
    if replace!=False:
        query = query.replace('_____',replace)
        
    cur.execute(query)
    results = cur.fetchall()
    
    if len(results) > 0:
        arr = np.array(results)
        df = pd.DataFrame(arr, columns=columns)
        return df
    
    else:
        df = pd.DataFrame()
        return df

def get_token(scope, client_id=st.secrets['info']['lightcast_client_id'], client_secret=st.secrets['info']['lightcast_api_key']): 
    
    # Connecting to API for OAuth2 token.
    # Setting up payload with id, secret, and scope.
    # Request "POST" response using the url.
    # Extract the token.
    
    url = "https://auth.emsicloud.com/connect/token"
    payload = "client_id={}&client_secret={}&grant_type=client_credentials&scope={}".format(client_id,client_secret,scope)
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.request("POST", url, data=payload, headers=headers)
    token = response.json()['access_token']
    return token

@st.cache_data
def generate_message(resume, jobs):

	return [{'role': 'system', 
             'content': f"""You are an assistant for a career coach, helping them prepare to lead a call with student who just graduated and is 
             preparing to enter the workforce. In 150 words or less, assess some of the strengths you see in this candidates resume. In addition 
             to this, in another 150 words or less, summarize how a candidate like this could position their experience for the following three 
             jobs that we've identified are a strong fit for them: {jobs}. Use a bullet point format. Here is the resume: {resume}"""
            }]

@st.cache_data
def chatgpt(message):
    while True:
        try:
            result = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=message)

            content = result['choices'][0]['message']['content']
            completion_tokens = result['usage']['completion_tokens']
            prompt_tokens = result['usage']['total_tokens']
            tokens = completion_tokens + prompt_tokens

            return content#, tokens
        # Lazy exception error handling to back off and try the API again.
        except:
                time.sleep(5)

# Resume to skill.
@st.cache_data
def resume_to_skills(text=None):
    
    skill_set = []
    
    # truncate text if needed.
    if len(text) > 50*(2**10):
        text = text[:50*(2**10)]
        # Print warning if text truncated.
        print('Warning: text too large. Truncating to first 50 kilobytes.')
    
    # Fetch Lightcast API. Pull out skills.
    token = get_token(scope='emsi_open')
    
    
    url = "https://emsiservices.com/skills/versions/latest/extract"
    headers = {'Authorization': 'Bearer {}'.format(token), 'content-type': 'text/plain'}
    payload = text
            
    r = requests.request("POST", url, data=payload.encode('UTF-8'), headers=headers)
    r = json.loads(r.text)
                
    try:
        skills_raw = r['data']
                
            
        for i in skills_raw:
            skill_set.append(i['skill']['name'])
    except:
            skill_set.append(None)
                
        
    return skill_set


## Establish Snowflake cursor.
cur = ctx.cursor()


# Fetch job skill relationships.
@st.cache_data
def job_skill_dataset():

	sql = """
	with sub as (
	select 
	    tj.name as job_name, 
	    tjp.median_salary,
	    tjp.unique_postings,
	    tjp.unique_companies,
	    ts.name as skill_name
	from 
	    discovery_pii.taxonomy_jobskills as tjs
	left join 
	    discovery_pii.taxonomy_job as tj
	on 
	    tjs.job_id = tj.id
	left join 
	    discovery_pii.taxonomy_skill as ts
	on 
	    tjs.skill_id = ts.id
	left join
		discovery_pii.taxonomy_jobpostings as tjp
	on
		tj.id = tjp.job_id

	qualify
	    row_number() over (partition by job_name order by ln(significance/2) * (tjs.unique_postings/1.5) desc) <= 15)

	select 
	    job_name, 
	    median_salary,
	    unique_postings,
	    unique_companies,
	    listagg(skill_name, ',')
	from 
	    sub
	group by 
	    1,2,3,4
	having
	    count(*) >= 10
	"""

	cols=['job_name','median_salary','unique_postings','unique_companies','skills']

	job_skills_mappings_df = run_query(query=sql, columns=cols)

	def listify(x):
		return x.split(',')

	job_skills_mappings_df['skills'] = job_skills_mappings_df['skills'].apply(listify)
	return job_skills_mappings_df

job_skills_mappings_df = job_skill_dataset()


@st.cache_data
def skill_metadata():
	sql = """
	select 
	    skill.name as skill,
	    skill.type_name as type,
	    subcategory.name as subcategory,
	    category.name as category,
	    skill.description
	from 
	    discovery_pii.taxonomy_skill as skill
	left join
	    discovery_pii.taxonomy_skillsubcategory as subcategory
	on
	    skill.subcategory_id = subcategory.id
	left join
	    discovery_pii.taxonomy_skillcategory as category
	on
	    skill.category_id = category.id
	"""

	cols = ['skill','type','subcategory','category','description']

	skills_dataframe = run_query(query=sql, columns=cols)

	return skills_dataframe

skills_dataframe = skill_metadata()


def jaccard_similarity(set_a, set_b):
    try:
        set_a = set(set_a)
        set_b = set(set_b)
        intersection = set_a.intersection(set_b)
        jaccard_similarity = len(intersection)/len(set_a.union(set_b))
        diff = set_b.difference(set_a)
        return jaccard_similarity, list(intersection), list(diff)
    except:
        return 0, [], []


@st.cache_data
def matching_jobs(skills,job_skills_mappings_df):

	job_name_list = []
	jaccard_list = []
	intersection_list = []
	difference_list = []

	for i in range(len(job_skills_mappings_df)):
		job_name = job_skills_mappings_df['job_name'][i]
		jaccard, intersection, difference = jaccard_similarity(set_a=skills, set_b=job_skills_mappings_df['skills'][i])
	    
		job_name_list.append(job_name)
		jaccard_list.append(jaccard)
		intersection_list.append(intersection)
		difference_list.append(difference)

	results_df = pd.DataFrame(data={
	    'job_name':job_name_list,
	    'jaccard':jaccard_list,
	    'intersection':intersection_list,
	    'difference':difference_list,
	    'unique_postings': [x for x in job_skills_mappings_df['unique_postings']],
	    'unique_companies': [x for x in job_skills_mappings_df['unique_companies']],
	    'median_salary': [x for x in job_skills_mappings_df['median_salary']]
	})

	results_df =results_df.sort_values(by='jaccard',ascending=False).reset_index().drop(columns=['index'])

	return results_df

@st.cache_data
def fetch_skill_metadata(skill):
	myskill = skills_dataframe[skills_dataframe['skill']==skill]
	return myskill

def annotations(skills, color):
    list_ = []
    for skill in skills:
        list_.append((skill,"",color))
        list_.append(', ')

    return list_[:-1]


st.title('Coaching Tool')
st.warning('Do not post any personally identifiable information about the learner here.', icon="⚠️")
with st.expander("How does this work?"):
    st.markdown("""
        Paste a resume for a student in the text field, then click cmd + enter. The following will be generated.
        (1) An interactive list of the skills found in the resume, with the ability to read more about them.
        (2) A sorted list of the jobs that intersect the most with the resume the learner has written.

        This was built as a prototype collaboration between Nadja Shaw from Workforce Engagement and Nathan Robertson 
        from Enterprise Platform Product.""")

st.subheader('Extract skills from a resume.')
txt = st.text_area('''
	Paste a resume here. 
	''',help='''
	The resume you paste here will be sent 
	to Lightcast to determine the skills it teaches.
	''',
	height=300)


if len(txt) >0:
	skills = resume_to_skills(txt)
	matching_jobs_df = matching_jobs(skills,job_skills_mappings_df)

	top_jobs = f'''{matching_jobs_df.iloc[0]['job_name']}, 
				   {matching_jobs_df.iloc[1]['job_name']}, and 
				   {matching_jobs_df.iloc[2]['job_name']}'''

	message = generate_message(resume=txt, jobs=top_jobs)
	response = chatgpt(message)

	st.subheader('ChatGPT Summary of Candidate\'s Resume')
	st.markdown('''_This text was generated by ChatGPT and should be reviewed carefully._''')
	st.write(f'''Below is a quick summary of the candidates strengths, 
		         and a few notes on how they could pursue a few jobs that
		         we thought were great matches for them based on the skills found
		         in their resume: {top_jobs}.''')

	st.markdown(response)


if len(txt) >0:
	st.subheader('Skills found in resume.')
	#skills = resume_to_skills(txt)
	#st.write(skills)

	selected = pills("Click a skill to learn more", skills)
	skill_metadata = fetch_skill_metadata(selected)

	try:
		skill_name = skill_metadata['skill'].iloc[0]
	except:
		skill_name = 'Unknown'
	try:
		subcategory_name = skill_metadata['subcategory'].iloc[0]
	except:
		subcategory_name = 'Unclassified Subcategory'
	try:
		category_name = skill_metadata['category'].iloc[0]
	except:
		category_name = 'Unclassified Category'
	try:
		description_text = skill_metadata['description'].iloc[0]
	except:
		description_text = 'There is no description for this skill.'

	st.subheader("What is {}?".format(skill_name))
	st.markdown("**Skill Taxonomy: {} -> {} -> {}**".format(category_name, subcategory_name, skill_name))
	st.markdown("*{}*".format(description_text))


if 'skills' in globals():

	st.subheader('Job Titles you Match')
	st.markdown('Ranked by how well your skills match the most in-demand skills for job titles.')
	st.markdown('---------')

	matching_jobs_df = matching_jobs(skills,job_skills_mappings_df)

	for i, row in matching_jobs_df[matching_jobs_df['jaccard']>0].iterrows():

		col1, col2 = st.columns([1,3])
		st.markdown('---------')
		#st.subheader('**{}**'.format(row['job_name']))
		with col1:

			st.subheader('**{}**'.format(row['job_name']))
			st.markdown('*Median US Salary: ${}*'.format(row['median_salary']))
			st.markdown('*Unique Companies Hiring: {}*'.format(row['unique_companies']))
		with col2:
			st.markdown('*Matching Skills:*')
			annots = annotations(skills=row['intersection'],color="#89CFF0")
			annotated_text(*annots)
			st.markdown('')
			st.markdown('*Missing Skills:*')
			annots = annotations(skills=row['difference'],color='#E5E4E2')
			annotated_text(*annots)
		
		



	#st.write(matching_jobs(skills,job_skills_mappings_df))

