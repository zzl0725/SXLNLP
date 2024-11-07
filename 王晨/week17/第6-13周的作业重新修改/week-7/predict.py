import torch
from config import Config
from model import TorchModel

def main():
    # 加载模型
    model = TorchModel(Config)
    model.load_state_dict(torch.load('./output/epoch_1.pth'))  # 加载权重
    # 加载测试数据
    test_data = [{"label":1,"review":"很好，送来的时候披萨还是烫的，送餐员很用心。"},
{"label":1,"review":"。送餐员特好的说话。。。。。。"},
{"label":0,"review":"基本不给低评价，这次是真怒了。早8点一过就下单晚到30分钟以上也就罢了，还联系不上，也不打电话说，孩子饿的不停叫唤。最后送来要的传统早餐套配错了，就一杯豆浆和咸菜。油条和豆腐脑不知所踪。也没让再补，因为已经10点了再送来就是中午饭了。真心不愉快。"},
{"label":0,"review":"没想到是甜的,不喜欢甜带鱼"},
{"label":0,"review":"里面的汤撒了"},
{"label":0,"review":"等了将近三个小时,饺子已经凉透了"},
{"label":0,"review":"冷了，饼太厚，口感很柴，锡箔纸包到饼内，很难撕开；,肘子，似乎不是肘子，是兽肉丁+肥肉丁"},
{"label":0,"review":"大酱汤和辣白菜五花肉饭蒜味真的太重了,吃着快吐了,少放点吧！桔梗特难吃。。"},
{"label":0,"review":"外卖餐盒还要钱不合理，而且餐盒很贵，几分钱的餐盒要一块一个"},
{"label":0,"review":"不知道是商家不做还是快递员不送，2个小时才送到，餐到已经彻底凉透，很是无语"},
{"label":0,"review":"确实还没煎饼摊儿上的好吃，优点是卫生吧。"},
{"label":0,"review":"东西跟图片完全不一样,鸡肉全是汤,卤肉也没几块肉渣,全是大白菜,极差评,太坑爹了"},
{"label":1,"review":"好极了！送来还是热的！"}]
    # 进行预测
    with torch.no_grad():
        outputs = model(test_data)
    # 输出预测结果
    print(outputs)
if __name__ == '__main__':
    main()