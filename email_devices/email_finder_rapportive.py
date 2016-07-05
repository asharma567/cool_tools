import subprocess

curl_auth_URL = '''
curl 'https://www.linkedin.com/uas/js/authuserspace?v=0.0.1191-RC8.56523-1429&api_key=4XZcfCb3djUl-DHJSFYd1l0ULtgSPl9sXXNGbTKT2e003WAeT6c2AqayNTIN5T1s&credentialsCookie=true' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36' -H 'Accept: */*' -H 'Referer: https://mail.google.com/mail/u/0/' -H 'Cookie: visit="v=1&M"; wutan=3tuIE4NFg+anLVR45/FRML6KkQZZREJ7voNVKBn5XwM=; bcookie="v=2&5119d04a-4d67-4d0f-8348-ac41c6529628"; bscookie="v=1&20160609204657aaeeccfd-80ab-4d70-80a0-cbb2736bed8eAQHhp-6ERxAp7k5vFJhfyGPoiYlBtxJ4"; _cb_ls=1; _chartbeat2=TZ1XNDgkXfwgrzHC.1465612818216.1466547387325.100101000001; oz_props_fetch_size1_112650289=15; share_setting=PUBLIC; sdsc=1%3A1SZM1shxDNbLt36wZwCgPgvN58iw%3D; _lipt=0_eXhgZe72hwJ1ACrCXPhsS-BdeFMrt54s5y8J_u6z8dzUxuf2h_YiYxmpomBbH-j5GxlEiquAYXl0GjOOBEqM3Bq_3Hi3HdpKqdeiuqMHz5h6CewjOlqYL9Qv3ejiVksrfRO1HnNsQ3vR_o0bKyts22tnnJkCZz7v1wGBmMVZmMpM7qwcXNsxZ3IMdxGqtFCJeNmC4SJGjzpxiWYPVGcE5dHsrhvqegaMtZ2s9dxg7gampby-4WXXuawXOcEJpJPmHD3yihk2-Snwob0GYdK1jMhQwJLxcE7WABN-Z_0z5MVzDSWNj5DScvLw1apCBPghb-ERr7LgtKmUV7ZUYMRhsXXpNcFXKU2FZzvIrRktts-; _ga=GA1.2.785583338.1465937479; lang="v=2&lang=en-us&c="; li_at=AQEDAQa26DEEU7yZAAABVVRR4zoAAAFVeryi4U0Ax6HKjV6DVd1MvPxJGttc4l48nYxa96oVVtytSm9pABwDDEJOk4oCqH_K2w17c5c-HdU5MhWFfLYCu9qcSdzadzddHTk5qru3fNUR6UcWRJDdBKAv; liap=true; sl="v=1&7dzNj"; JSESSIONID="ajax:1898019488492729741"; lidc="b=LB89:g=357:u=181:i=1466637083:t=1466715123:s=AQEs-9w5sEyrnjA_QAFN9SCQxeKHkE5H"' -H 'Connection: keep-alive' --compressed
'''
curl_profile_URL = '''
curl 'https://api.linkedin.com/v1/people/email={0}%40{1}:(first-name,last-name,headline,location,distance,positions,twitter-accounts,im-accounts,phone-numbers,member-url-resources,picture-urls::(original),site-standard-profile-request,public-profile-url,relation-to-viewer:(connections:(person:(first-name,last-name,headline,site-standard-profile-request,picture-urls::(original)))))' -H 'Cookie: bcookie="v=2&5119d04a-4d67-4d0f-8348-ac41c6529628"; sdsc=1%3A1SZM1shxDNbLt36wZwCgPgvN58iw%3D; _lipt=0_eXhgZe72hwJ1ACrCXPhsS-BdeFMrt54s5y8J_u6z8dzUxuf2h_YiYxmpomBbH-j5GxlEiquAYXl0GjOOBEqM3Bq_3Hi3HdpKqdeiuqMHz5h6CewjOlqYL9Qv3ejiVksrfRO1HnNsQ3vR_o0bKyts22tnnJkCZz7v1wGBmMVZmMpM7qwcXNsxZ3IMdxGqtFCJeNmC4SJGjzpxiWYPVGcE5dHsrhvqegaMtZ2s9dxg7gampby-4WXXuawXOcEJpJPmHD3yihk2-Snwob0GYdK1jMhQwJLxcE7WABN-Z_0z5MVzDSWNj5DScvLw1apCBPghb-ERr7LgtKmUV7ZUYMRhsXXpNcFXKU2FZzvIrRktts-; _ga=GA1.2.785583338.1465937479; lang="v=2&lang=en-us&c="; liap=true; lidc="b=LB89:g=357:u=181:i=1466635831:t=1466715123:s=AQH5lo0WCPxkRJs4xxddhuYc__WN3Xh4"' -H 'X-Cross-Domain-Origin: https://mail.google.com' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36' -H 'Content-type: application/json' -H 'Accept: */*' -H 'Referer: https://api.linkedin.com/uas/js/xdrpc.html?v=0.0.1191-RC8.56523-1429' -H 'X-Requested-With: IN.XDCall' -H 'x-li-format: json' -H 'Connection: keep-alive' -H 'oauth_token: {2}' -H 'X-HTTP-Method-Override: GET' --compressed
'''

string_formats = '''
	{first_name}@{domain}
	{last_name}@{domain}
	{first_name}{last_name}@{domain}
	{first_name}.{last_name}@{domain}
	{f_initial}{last_name}@{domain}
	{f_initial}.{last_name}@{domain}
	{first_name}{l_initial}@{domain}
	{first_name}.{l_initial}@{domain}
	{f_initial}{l_initial}@{domain}
	{f_initial}.{l_initial}@{domain}
	{last_name}{first_name}@{domain}
	{last_name}.{first_name}@{domain}
	{last_name}{f_initial}@{domain}
	{last_name}.{f_initial}@{domain}
	{l_initial}{first_name}@{domain}
	{l_initial}.{first_name}@{domain}
	{l_initial}{f_initial}@{domain}
	{l_initial}.{f_initial}@{domain}
	{f_initial}{last_name}@{domain}
	{f_initial}.{last_name}@{domain}
	{first_name}{last_name}@{domain}
	{first_name}..{last_name}@{domain}
	{first_name}{last_name}@{domain}
	{first_name}..{last_name}@{domain}
	{first_name}-{last_name}@{domain}
	{f_initial}-{last_name}@{domain}
	{first_name}-{l_initial}@{domain}
	{f_initial}-{l_initial}@{domain}
	{last_name}-{first_name}@{domain}
	{last_name}-{f_initial}@{domain}
	{l_initial}-{first_name}@{domain}
	{l_initial}-{f_initial}@{domain}
	{f_initial}-{last_name}@{domain}
	{first_name}--{last_name}@{domain}
	{first_name}--{last_name}@{domain}
	{first_name}_{last_name}@{domain}
	{f_initial}_{last_name}@{domain}
	{first_name}_{l_initial}@{domain}
	{f_initial}_{l_initial}@{domain}
	{last_name}_{first_name}@{domain}
	{last_name}_{f_initial}@{domain}
	{l_initial}_{first_name}@{domain}
	{l_initial}_{f_initial}@{domain}
	{f_initial}_{last_name}@{domain}
	{first_name}__{last_name}@{domain}
	{first_name}__{last_name}@{domain}
'''



def curl_this(url):
    '''
    I: string, prepended by 'curl'
    O: output string of curl
    '''

    stor = subprocess.Popen(
        url,
        shell=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    ).communicate()
    return stor


def parse_auth_token(curled_output):
	#build regex or something to parse this 
    if curled_output[0] != '':
        cut_off_front = curled_output[0].split('l.oauth_token =')[1] 
        cut_off_back = cut_off_front.split(';')[0]
        auth_token_str = cut_off_back.strip().strip('\"')
    else:
        print output[1]
        return None
    return auth_token_str


def get_auth_token(URL):
    return parse_auth_token(curl_this(URL))

def get_profile_from_LI(user_handle, domain, auth_token):
    profile = curl_this(curl_profile_URL.format(user_handle, domain, auth_token))
    return profile

def check_LI_profile_exist(user_handle, domain, auth_token):
    #hacky ass way to this
    output_str = get_profile_from_LI(user_handle,domain, auth_token)
    #put an if clause and parse out the messsage. 
    #it's the token call this same function again.
    return  ('''"message": "Couldn\'t find member''' not in output_str[0])

#I, O first name, last name, domain
def permuter(first_name, last_name, domain):
    disc = {
        'first_name': first_name, 
        'last_name': last_name,
        'f_initial': first_name[0], 
        'l_initial': last_name[0],
        'domain': domain,
    }
    
    emails = [template.format(**disc).strip() for template in string_formats.split('\n')]
    return emails[1:-1]


def email_checker(first_name, last_name, domain):
    # it's not very fast so lets multithread
    auth_token = get_auth_token(curl_auth_URL)
    list_of_emails = permuter(first_name, last_name, domain)

    output = []    
    for email in list_of_emails:
        user_handle, domain = email.split('@')
        boolean = check_LI_profile_exist(user_handle, domain, auth_token)
        output.append((email, boolean))
    return output

if __name__ == '__main__':
	
	    # ('joe','choti', 'mlb.com'),
	    # ('laura', 'kang', 'sendence.com'),
	   	# ('rasmus', "wissmann",  'oscarhealth.com'),
	    # ('paolo', 'esquivel', 'truveris.com'),
	    # ('carl', 'anderson', 'wework.com'),
	    # ('carl', 'vogel', 'warbyparker.com'),
	    # ('max', 'shron', 'warbyparker.com'),

	# some_list = [
	#     ('Debbie', 'Chung', 'gilt.com'),
	#     ('Igor', 'Elbert', 'gilt.com'),
	#     ('asharma','567567', 'gmail.com'),
	#     ('claudia', 'perlich', 'dstillery.com'),
	#     ('brian', 'dalessandro', 'zocdoc.com'),
	#     ('jeffrey', 'picard', 'contently.com'),
	# ]
	str_lis = '''
		brian dalessandro zocdoc.com
		Dan Becker datarobot.com
		Satadru Sengupta datarobot.com
		xavier datarobot.com
		Srikesh Arunajadai  Celmatix.com
		Bob Sohval  securityscorecard.io
		ppoh securityscorecard.io
		Richard L Williams makerbot.com
		Russell Kummer makerbot.com
		Alexandra Marvar makerbot.com
		Jack Thompson makerbot.com
		Jonathan Taqqu intentmedia.com
		careers sendence.com
		Laura sendence.com
	'''
	'''
	brian@zocdoc.com
	'''
	some_list = [tuple(item.strip('\t').split()) for item in str_lis.split('\n') if item != '']

	for args in some_list:
		bag = []
		
		for email in email_checker(*args):
			try:
				if email[1]: bag.append(email_checker(*args))
			except: 
				print email
		
		
		
	if bag:
		print '\n'.join(map(str,bag[::-1]))
	else:
		print None

