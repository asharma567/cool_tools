import subprocess

#we could also break this up since we know the test string works
auth_URL = '''
'https://www.linkedin.com/uas/js/authuserspace?v=0.0.1191-RC8.56523-1429&api_key=4XZcfCb3djUl-DHJSFYd1l0ULtgSPl9sXXNGbTKT2e003WAeT6c2AqayNTIN5T1s&credentialsCookie=true' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36' -H 'Accept: */*' -H 'Referer: https://mail.google.com/mail/u/0/' -H 'Cookie: visit="v=1&M"; wutan=3tuIE4NFg+anLVR45/FRML6KkQZZREJ7voNVKBn5XwM=; bcookie="v=2&5119d04a-4d67-4d0f-8348-ac41c6529628"; bscookie="v=1&20160609204657aaeeccfd-80ab-4d70-80a0-cbb2736bed8eAQHhp-6ERxAp7k5vFJhfyGPoiYlBtxJ4"; _cb_ls=1; _chartbeat2=TZ1XNDgkXfwgrzHC.1465612818216.1466547387325.100101000001; oz_props_fetch_size1_112650289=15; share_setting=PUBLIC; sdsc=1%3A1SZM1shxDNbLt36wZwCgPgvN58iw%3D; _lipt=0_eXhgZe72hwJ1ACrCXPhsS-BdeFMrt54s5y8J_u6z8dzUxuf2h_YiYxmpomBbH-j5GxlEiquAYXl0GjOOBEqM3Bq_3Hi3HdpKqdeiuqMHz5h6CewjOlqYL9Qv3ejiVksrfRO1HnNsQ3vR_o0bKyts22tnnJkCZz7v1wGBmMVZmMpM7qwcXNsxZ3IMdxGqtFCJeNmC4SJGjzpxiWYPVGcE5dHsrhvqegaMtZ2s9dxg7gampby-4WXXuawXOcEJpJPmHD3yihk2-Snwob0GYdK1jMhQwJLxcE7WABN-Z_0z5MVzDSWNj5DScvLw1apCBPghb-ERr7LgtKmUV7ZUYMRhsXXpNcFXKU2FZzvIrRktts-; _ga=GA1.2.785583338.1465937479; lang="v=2&lang=en-us&c="; li_at=AQEDAQa26DEEU7yZAAABVVRR4zoAAAFVeryi4U0Ax6HKjV6DVd1MvPxJGttc4l48nYxa96oVVtytSm9pABwDDEJOk4oCqH_K2w17c5c-HdU5MhWFfLYCu9qcSdzadzddHTk5qru3fNUR6UcWRJDdBKAv; liap=true; sl="v=1&7dzNj"; JSESSIONID="ajax:1898019488492729741"; lidc="b=LB89:g=357:u=181:i=1466637083:t=1466715123:s=AQEs-9w5sEyrnjA_QAFN9SCQxeKHkE5H"' -H 'Connection: keep-alive' --compressed
'''

try_this = ''' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36' -H 'Accept: */*' -H 'Referer: https://mail.google.com/mail/u/0/' -H 'Cookie: visit="v=1&M"; wutan=3tuIE4NFg+anLVR45/FRML6KkQZZREJ7voNVKBn5XwM=; bcookie="v=2&5119d04a-4d67-4d0f-8348-ac41c6529628"; bscookie="v=1&20160609204657aaeeccfd-80ab-4d70-80a0-cbb2736bed8eAQHhp-6ERxAp7k5vFJhfyGPoiYlBtxJ4"; _cb_ls=1; _chartbeat2=TZ1XNDgkXfwgrzHC.1465612818216.1466547387325.100101000001; oz_props_fetch_size1_112650289=15; share_setting=PUBLIC; sdsc=1%3A1SZM1shxDNbLt36wZwCgPgvN58iw%3D; _lipt=0_eXhgZe72hwJ1ACrCXPhsS-BdeFMrt54s5y8J_u6z8dzUxuf2h_YiYxmpomBbH-j5GxlEiquAYXl0GjOOBEqM3Bq_3Hi3HdpKqdeiuqMHz5h6CewjOlqYL9Qv3ejiVksrfRO1HnNsQ3vR_o0bKyts22tnnJkCZz7v1wGBmMVZmMpM7qwcXNsxZ3IMdxGqtFCJeNmC4SJGjzpxiWYPVGcE5dHsrhvqegaMtZ2s9dxg7gampby-4WXXuawXOcEJpJPmHD3yihk2-Snwob0GYdK1jMhQwJLxcE7WABN-Z_0z5MVzDSWNj5DScvLw1apCBPghb-ERr7LgtKmUV7ZUYMRhsXXpNcFXKU2FZzvIrRktts-; _ga=GA1.2.785583338.1465937479; lang="v=2&lang=en-us&c="; li_at=AQEDAQa26DEEU7yZAAABVVRR4zoAAAFVeryi4U0Ax6HKjV6DVd1MvPxJGttc4l48nYxa96oVVtytSm9pABwDDEJOk4oCqH_K2w17c5c-HdU5MhWFfLYCu9qcSdzadzddHTk5qru3fNUR6UcWRJDdBKAv; liap=true; sl="v=1&7dzNj"; JSESSIONID="ajax:1898019488492729741"; lidc="b=LB89:g=357:u=181:i=1466637083:t=1466715123:s=AQEs-9w5sEyrnjA_QAFN9SCQxeKHkE5H"' -H 'Connection: keep-alive' --compressed'''

out_url = 'https://www.linkedin.com/uas/js/authuserspace?v=0.0.1191-RC8.56523-1429&api_key=4XZcfCb3djUl-DHJSFYd1l0ULtgSPl9sXXNGbTKT2e003WAeT6c2AqayNTIN5T1s&credentialsCookie=true'

if __name__ == '__main__':

	
	# print auth_URL
	# print '=' * 50
	# print out_url + try_this
	
	all = out_url + try_this

	stor = subprocess.Popen(
	    ['curl', all.strip('\n')],
	    stdout=subprocess.PIPE, 
	    stderr=subprocess.PIPE
	).communicate()

	print stor