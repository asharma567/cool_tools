'''
Dependencies

* sshpass: bash program which allows you to bypass the login sequence for ssh
* subprocess: pip installable package
* pandas

This should definitely be written in OOP style. (pending)
'''

import pandas as pd
import subprocess

EXAMPLE_QUERY_STR = '''
SET hive.cli.print.header=true;
SELECT 
    global_property_id,
    CASE WHEN seller_designation = 'Owner-Manager' THEN 1 ELSE 0 END as is_owner
FROM derived.asset_commercial
WHERE is_auctioned = 1
;
'''

USER_NAME_ON_GATEWAY_MACHINE = 'your_user_name'
PW = 'your_password'


GATEWAY_LOGIN = [
    'sshpass',
    '-p',
    PW,
    "ssh",
    USER_NAME_ON_GATEWAY_MACHINE +'@hgat01lax01us.prod.auction.local'
    ]

QUERY_PREFIX = [
    'hive', 
    '-S', 
    '-e', 
    '\"', 
]

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def hive_query_to_df(query_str):
    # NOTE can't send in query without the names of the field just yet
    '''
    I: query string eg 'SELECT field1, field2 FROM my_hdfs_table;'
    O: pandas dataframe
    '''
    query_str = query_str.replace('"', '\'')
    #reading the query splitting it into word for items in a list
    list_of_split_lines = [line.strip().split() for line in query_str.split('\n')[1:-1]]
    
    #unpacking the list
    query_list = [sub_list for lists in list_of_split_lines for sub_list in lists]
    query = QUERY_PREFIX + query_list + ['\"']

    #calling the function
    proc = subprocess.Popen(GATEWAY_LOGIN + query, stdout=subprocess.PIPE)
    stdout_value = proc.communicate()[0]

    #parsing the output
    raw_output_data = [line.split('\t') for line in stdout_value.split('\n')]
    
    #extracting the header
    header = raw_output_data[0]
    df = pd.DataFrame(raw_output_data[1:-1], columns=header)
    
    #convert the columns back to their proper data type
    df = df.applymap(lambda x: float(x) if isfloat(x) else x)
    df = df.applymap(lambda x: int(x) if isint(x) else x)
    
    return df


if __name__ == '__main__':


    print hive_query_to_df(EXAMPLE_QUERY_STR)