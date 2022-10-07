# from __future__ import division
import itertools

# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os
from scipy.stats import spearmanr
from sklearn import metrics
from random import randint

# Custom Module Imports
import env

def get_data():
    """
    This function acquires the data from the codeup database
    and save it into a dataframe.
    """
    url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/curriculum_logs'
    query = '''
    SELECT *
    FROM logs
    LEFT JOIN cohorts ON logs.cohort_id=cohorts.id;
    '''
    df = pd.read_sql(query, url)
    return df

def acquire():
    filename = "curriculum_logs.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = get_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 

def wrangle(df):
    """
    This function wrangles the data for exploration by filling nulls, 
    dropping unneeded columns, creating new columns, and setting datetimes.
    """
    # Dropping unneeded columns
    df = df.drop(columns=['deleted_at', 'updated_at', 'created_at', 'slack'])

    # Filling cohort_id nulls with 0
    df['cohort_id'] = df['cohort_id'].fillna(0.0)

    # Removing the root path
    df = df[df['path'] != '/']

    # Creating columns with new splits in the path. This helps identify lessons easier.
    str_split = df.path.str.split('/', expand=True)
    str_split = str_split.drop(columns=[2,3,4,5,6,7])
    str_split = str_split.dropna(axis=0)
    root = str_split[0]
    root = pd.DataFrame(root)
    df = pd.merge(df, root, how='left', left_index=True, right_index=True)
    df.rename(columns={0:'root_path'}, inplace=True)
    root2 = str_split[0] + '/' + str_split[1]
    root2 = pd.DataFrame(root2)
    df = pd.merge(df, root2, how='left', left_index=True, right_index=True)
    df.rename(columns={0:'root_path_2'}, inplace=True)

    # Setting Datetimes
    df.date = pd.to_datetime(df.date)
    df.start_date = pd.to_datetime(df.start_date)
    df.end_date = pd.to_datetime(df.end_date)

    return df

def q1(df):
    """
    This function returns two dataframes for pages by cohort and max page by cohort.
    """
    # Groupby the cohort_id and the path
    page_by_cohort = df.groupby(['cohort_id'])['path'].value_counts()
    # convert into a dataframe
    page_by_cohort = pd.DataFrame(page_by_cohort)
    # rename the column
    page_by_cohort.columns=['path_value_count']
    # reset the index
    page_by_cohort = page_by_cohort.reset_index()
    # create dataframe with max per cohort id
    max_page_by_cohort = page_by_cohort.groupby('cohort_id').max()
    # set the index
    page_by_cohort = page_by_cohort.set_index('cohort_id')

    return page_by_cohort, max_page_by_cohort

def q3(df):
    """
    This function returns a dataframe that takes the value count of the root_path_2
    from a dataframe without staff included.
    """
    # Make a dataframe that doesn't include staff
    df_without_staff = df[df['cohort_id'] != 28.0]
    # Get the value count for the root_path_2
    least_lessons_without_staff = df_without_staff.root_path_2.value_counts().sort_values()
    # Turn it into a dataframe
    least_lessons_without_staff = pd.DataFrame(least_lessons_without_staff)
    
    return least_lessons_without_staff

def q4(df):
    """
    This function returns 2 dataframes that contain the most accessed 
    lessons after graudation by program.
    """
    # create a dataframe for dates after graduation
    after_grad_df = df[df.end_date < df.date]
    # Create a dataframe that groups the program id and path
    page_by_program = after_grad_df.groupby(['program_id'])['path'].value_counts()
    # convert to dataframe
    page_by_program = pd.DataFrame(page_by_program)
    # rename the column
    page_by_program.columns=['path_value_count']
    # reset the index
    page_by_program = page_by_program.reset_index()
    # make a dataframe that groups program id by max
    max_page_by_program = page_by_program.groupby('program_id').max()
    # set the index
    page_by_program = page_by_program.set_index('program_id')

    return page_by_program, max_page_by_program

def q5(df):
    """
    This function returns a dataframe that groups by user id and root path.
    """
    # creates a dataframe that pulls dates in between the start and end date
    active_df = df[(df.end_date >= df.date) & (df.start_date <= df.date)]
    # creates a dataframe that groups by user id and root path
    page_by_student = active_df.groupby(['user_id'])['root_path'].value_counts()
    # convert to dataframe
    page_by_student = pd.DataFrame(page_by_student)
    # rename the column
    page_by_student.columns=['root_path_value_count']
    # reset the index
    page_by_student = page_by_student.reset_index()
    # group by user id to get the sum of the root_path_value_count
    page_by_student = page_by_student.groupby('user_id').sum('root_path_value_count')

    return page_by_student, active_df