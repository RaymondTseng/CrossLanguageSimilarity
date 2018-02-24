import requests
from lxml import etree

# head_url = 'http://es.euronews.com'
# start_url = 'http://es.euronews.com/european-affairs'
# head_url = 'http://spanish.xinhuanet.com'
# start_url = 'http://spanish.xinhuanet.com'
head_url = 'https://www.spanishnews.es'
start_url = 'https://www.spanishnews.es'
save_path = '/home/raymond/Downloads/spanish_news/'


# visited_url_set = set()
# news_url_set = set()
# unvisit_url_set = set()
#
# unvisit_url_set.add(start_url)
#
# while len(unvisit_url_set) != 0:
#     url = unvisit_url_set.pop()
#     visited_url_set.add(url)
#     try:
#         response = requests.get(url)
#         html = etree.HTML(response.text)
#         results = html.xpath('//a/@href')
#         for t in results:
#             if not t.startswith(head_url):
#                 continue
#             if t not in visited_url_set:
#                 if t not in unvisit_url_set:
#                     unvisit_url_set.add(t)
#                     if t.startswith(head_url + '/20'):
#                         news_url_set.add(t)
#                         print('news:' + t)
#     except:
#         print('error:' + url)
count = 9250
f = open('temp1.txt', 'r')
for line in f.readlines():
    line = line.strip()
    if line.startswith('news:'):
        url = line.split('news:')[1]
    else:
        continue
    print(url)
    try:
        response = requests.get(url)
    except:
        print('error!!')
        continue
    html = etree.HTML(response.text)
    try:
        title = html.xpath('//span[@id="bltitle"]/text()')[0].encode('utf-8').strip()
    except:
        title = ''

    if not title:
        try:
            title = html.xpath('//div[@class="bt"]/h2/text()')[0].encode('utf-8').strip()
        except:
            title = ''
    if not title:
        try:
            title = html.xpath('//div[@id="Title"]/text()')[0].encode('utf-8').strip()
        except:
            title = ''
    if not title:
        try:
            title = html.xpath('//span[@id="whtitle"]/text()')[0].encode('utf-8').strip()
        except:
            title = ''
    try:
        content = html.xpath('//span[@id="content"]//p/text()')
    except:
        content = []

    if len(content) == 0:
        try:
            content = html.xpath('//div[@class="nr"]/font//p/text()')
        except:
            content = []

    if len(content) == 0:
        try:
            content = html.xpath('//div[@id="Content"]/font//p/text()')
        except:
            content = []
    print(content)

    if not title and len(content) == 0:
        continue

    wf = open(save_path + str(count) + '.txt', 'wr')
    wf.write(title + '\n')
    # wf.write(summary + '\n')
    for c in content:
        wf.write(c.encode('utf-8').strip() + '\n')
    wf.close()
    count += 1

f.close()