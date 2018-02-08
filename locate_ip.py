import scrapy


class IPCamSpider(scrapy.Spider):
    name = "IP_cam_getter"

    start_urls = []
    for idx in range(1, 31):
        start_urls.append('https://www.insecam.org/en/bytag/Traffic/?page=%s' % idx)

    def parse(self, response):
        for page in response.css('div.col-xs-12 col-sm-6 col-md-4 col-lg-4'):
            print(page)


a = IPCamSpider()
