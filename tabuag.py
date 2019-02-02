import datetime
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from numpy.matlib import rand
from matplotlib.mlab import dist
from matplotlib.artist import getp
import copy
from cProfile import run

#导入城市信息（data frame）
def_citieslist=pd.read_csv('F:\iwidev\iwidev2\city1.csv')

#城市信息，相关结构（城市索引，（纬度，经度））
city=[]
city_x = []
city_y = []
#解析数据框，按索引和行解析
for index,row in def_citieslist.iterrows():
	city.extend([(row[0],(row[1],row[2]))])
	
#城市数量
num=len(city)
distance = [[0 for col in range(num)] for raw in range(num)]

#禁忌表
tabu_list = []
tabu_time = []
#当前禁忌对象数量
current_tabu_num = 0

#禁忌长度，即禁忌期限
tabu_limit = 50

#单次迭代所得的领域解的个数
search_mun=0

#初始解
initial_route=[]
initial_distance=0.0

#候选集
candidate = []
candidate_distance = []

best_route = []
best_distance = sys.maxsize
current_route = []
current_distance = 0.0

def printline(city):
	for j in range(len(city)-1):
		print(city[j],city[j+1])
	print(city[len(city)],city[0])
#测试数据源导入结果
#printline(city)

def getdistancematrix():
	global city_x
	global city_y
	#构建经纬度列表
	for i in range(len(city)):
		city_x.append(city[i][1][0])
		city_y.append(city[i][1][1])
	
	for i in range(len(city)):
		for j in range (len(city)):
			dis=((city_x[i]-city_x[j])**2+(city_y[i]-city_y[j])**2)**0.5
			if dis==0:
				dis=sys.maxsize
			distance[i][j]=dis
	#print(distance)
	
#getdistancematrix()

#计算已知行进顺序的城市路线
def calculatetotalcost(list):
	cost=0
	for i in range(len(list)-1):
		cost=cost+((float(list[i][1][0])-float(list[i+1][1][0]))**2+(float(list[i][1][1])-float(list[i+1][1][1]))**2)**0.5
	cost=cost+((float(list[-1][1][0])-float(list[0][1][0]))**2+(float(list[-1][1][1])-float(list[0][1][1]))**2)**0.5
	# print(range(len(list)-1))
	print(cost)
	return (cost)

#根据已知路径利用距离矩阵计算移动代价
def calculatecostwithknownroute(route):
	totalcost=0.0
	for i in range(num-1):
		totalcost+=distance[route[i]][route[i+1]]
	totalcost+=distance[route[num-1]][route[0]]
	return totalcost
#calculatetotalcost(city)

# 通过贪婪算法确定初始解
def greedyforinitialsolution():
	# 初始化总距离赋值
	sum = 0.0
	# 构建各个城市间的距离坐标矩阵（对角矩阵）
	distancematrix = [[0 for col in range (num)] for raw in range (num)]
	for i in range(num):
		for j in range ( num ):
			distancematrix[i][j]=distance[i][j]
	visited=[]
	# 进行贪婪选择——每次都选择距离最近的
	id = 0
	for i in range (num):
		for j in range(num):
			#防止经过的城市再次被选为目标，设置无限大阻断值
			distancematrix[j][id]=sys.maxsize
		#取该数组的最小值
		minvalue = min (distancematrix[id])
		if i != num:
			#各数组最小值累加
			sum += minvalue
		#记录各城市，到其他最小距离城市的id（城市唯一标识）
		visited.append (id)
		#上一次移动的终点作为下一次移动的起点
		id = distancematrix[id].index(minvalue)
	#更新原有的距离矩阵（第一座城市到最后一座城市的距离）
	sum += distance[0][visited[num-1]]
	# print(visited)
	return visited
# greedyforinitialsolution()

#初始参数设置
def setparam():
	global best_route
	global best_distance
	global tabu_time
	global current_tabu_num
	global current_distance
	global current_route
	global tabu_list
	global initial_route
	global initial_distance
	#获得初始解
	current_route=greedyforinitialsolution()
	#记录初始解路径
	initial_route=copy.copy(current_route)
	#记录当前最优路径
	best_route = copy.copy (current_route)
	#记录当前移动成本
	current_distance=calculatecostwithknownroute(current_route)
	#初始解替换最优路程
	best_distance=current_distance
	#记录初始解成本
	initial_distance=copy.copy(current_distance)
	
	#清空禁忌表
	tabu_list.clear()
	tabu_time.clear()
	current_tabu_num=0
	
#交换当前解（数组）中的两个元素
def exchangeelement(index1,index2,list):
	currentlist=copy.copy(list)
	currentelement=currentlist[index1]
	currentlist[index1]=currentlist[index2]
	currentlist[index2]=currentelement
	return currentlist

#得到邻域 候选解
def get_candidate():
    global best_route
    global best_distance
    global current_tabu_num
    global current_distance
    global current_route
    global tabu_list
    global candidate
    global candidate_distance
    
    #存储两个交换的位置
    exchange_position = []
    temp = 0
    #候选集
    candidate = [[0 for col in range ( num )] for raw in range ( search_mun )]
    candidate_distance = [0 for col in range ( search_mun )]
    #随机选取邻域
    while True:
	    #在已有路径上随机选取两个城市交换顺序
        current = random.sample(range(0, num), 2)
        #判断交换方案是否近期已经实施，未实施才可进行交换
        if current not in exchange_position:
            exchange_position.append(current)
            candidate[temp] = exchangeelement(current[0], current[1], current_route)
            #获得新解后 检查是否在禁忌表内出现，控制迭代次数为10次
            if candidate[temp] not in tabu_list:
                candidate_distance[temp] = calculatecostwithknownroute(candidate[temp])
                temp += 1
            if temp >= search_mun:
                break
    
    #获得候选解中的最优解
    candidate_best = min(candidate_distance)
    best_index = candidate_distance.index(candidate_best)
    
    current_distance = candidate_best
    current_route = copy.copy (candidate[best_index])

    # 领域最优解和当前解作比较
    if current_distance < best_distance:
	    best_distance = current_distance
	    best_route = copy.copy ( current_route )

    # 进入禁忌表
    tabu_list.append(candidate[best_index])
    tabu_time.append(tabu_limit)
    current_tabu_num+=1

#更新禁忌表和禁忌期限
def update_tabulist():
	global tabu_list
	global tabu_time
	global current_tabu_num
	
	del_num=0
	temp = [0 for col in range (num)]
	
	# 更新步长(递归，每迭代一次，禁忌表可用空间减少一次)
	tabu_time = [x-1 for x in tabu_time]
	# 达到期限，释放元素，禁忌表满（可禁忌次数剩余0），则删除一个元素
	for i in range(current_tabu_num):
		if tabu_time[i]==0:
			del_num+=1
			tabu_list[i] = temp
	
	current_tabu_num -= del_num
	while 0 in tabu_time:
		tabu_time.remove ( 0 )
	
	while temp in tabu_list:
		tabu_list.remove (temp)
	
def update_tabu_list():
	global tabu_list
	global current_tabu_num
	
	if current_tabu_num==tabu_limit:
		tabu_list.remove(tabu_list[0])
		current_tabu_num-=1


def visualization():
	global city_x
	global city_y
	result_x = [0 for col in range ( num + 1 )]
	result_y = [0 for col in range ( num + 1 )]
	
	for i in range ( num ):
		result_x[i] = city_x[best_route[i]]
		result_y[i] = city_y[best_route[i]]
	result_x[num] = result_x[0]
	result_y[num] = result_y[0]
	print("最优路径对应城市坐标：")
	for j in range(num):
		print("("+str(result_x[j])+","+str(result_y[j])+")")
	# print ( result_x )
	# print ( result_y )
	plt.xlim ( 0, 2000 )  # 限定横轴的范围
	plt.ylim ( 0, 2500 )  # 限定纵轴的范围
	plt.plot ( result_x, result_y, marker='>', mec='r', mfc='w', label=u'Route' )
	plt.legend ( )  # 让图例生效
	plt.margins ( 0 )
	plt.subplots_adjust ( bottom=0.15 )
	plt.xlabel ( u"x" )  # X轴标签
	plt.ylabel ( u"y" )  # Y轴标签
	plt.title ( "TSP Solution" )  # 标题
	plt.show ( )
	plt.close ( 0 )

#求解
def solve():
	global  search_mun
	getdistancematrix ( )
	search_mun=int(input ( "单次迭代所得领域解的个数：" ))
	runtime = int (input ( "迭代次数：" ) )
	print("正在初始化参数...")
	setparam ( )
	print("参数初始化完成...")
	Starttime=datetime.datetime.now()
	print (str(Starttime))
	print("程序开始执行..." )
	for rt in range ( runtime ):
		print("第"+str(rt+1)+"次迭代开始")
		get_candidate ( )
		update_tabulist ( )
		print("第"+str(rt+1)+"次迭代结束")
	print("初始距离：")
	print(initial_distance)
	print("初始路径：")
	print(initial_route)
	print("当前距离：")
	print(current_distance)
	print("当前路径：")
	print(current_route)
	print( "最优距离：" )
	print( best_distance )
	print( "最优路径：" )
	print(best_route)
	visualization ()
	Endtime = datetime.datetime.now ( )
	print(str(Endtime))
	print("程序执行成功..." )
	print("本次计算总共耗时："+str(Endtime-Starttime)+" 秒")

solve()


# tabu_list.append([0])
# current_tabu_num=1
# update_tabu_list()
# tabu_list.append([1])
# current_tabu_num=2
# update_tabu_list()
# tabu_list.append([2])
# current_tabu_num=3
# update_tabu_list()
# tabu_list.append([3])
# current_tabu_num=4
# update_tabu_list()
# tabu_list.append([4])
# current_tabu_num=5
# update_tabu_list()
# tabu_list.append([5])
# current_tabu_num=5
# update_tabu_list()
# tabu_list.append([6])
# current_tabu_num=5
# update_tabu_list()
# tabu_list.append([7])
# current_tabu_num=5
# update_tabu_list()

# tabu_list.append([0])
# current_tabu_num=1
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([1])
# current_tabu_num=2
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([2])
# current_tabu_num=3
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([3])
# current_tabu_num=4
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([4])
# current_tabu_num=5
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([5])
# current_tabu_num=5
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([6])
# current_tabu_num=5
# tabu_time.append(5)
# update_tabulist()
# tabu_list.append([7])
# current_tabu_num=5
# tabu_time.append(5)
# update_tabulist()






# def draw():
# 	current = random.sample ( range ( 0, num ), 2 )
# 	print ( current )
#
# # draw()
# print (tabu_time)