import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import time


def dataimport(name):
	t1 = open("mushroom.dat", "r")
	#t1 =open("foodmartFIM.txt",'r')
	reader = t1.read().rstrip(" ").split("\n")
	size = len(reader)
	context = []
	for line in range(size):
		temp = reader[line].rstrip(" ").split(" ")
		context.append(temp)
	t1.close()
	return context


def buildmatrix(data, sup,datalength):
	# 单项统计字典，便于根据支持度删值
	dictItem = {}
	for items in data:
		for item in items:
			if str(item) in dictItem.keys():
				dictItem[str(item)] += 1
			else:
				dictItem[str(item)] = 1

	list_item_d = {}
	for key in dictItem.keys():
		if dictItem[key] >= sup:
			list_item_d[key]=dictItem[key]
	list_item = list(list_item_d.keys())
	dictM = {}
	for i in range(len(data)):
		try:
			dictM[tuple(data[i])] += 1
		except KeyError:
			dictM[tuple(data[i])] = 1
	list_keys = list(dictM.keys())
	list_ae = list(dictM.values())

	matrix = np.zeros((len(list_item), len(list_ae)))
	for i in range(len(list_keys)):
		for j in range(len(list_keys[i])):
			if list_keys[i][j] in list_item:
				n = list_item.index(list_keys[i][j])
				matrix[n][i] = 1

	df = DataFrame(matrix, index=list_item)

	num = df.shape[1]
	array = [x for x in range(num)]
	df1 = df.sort_values(by=array)
	list_length = list(np.sum(df))
	list_item = list(df.index)
	df2 = np.array(df1)
	return df2, list_item_d, list_item,list_ae,list_length

'''def dictquicksort(dictM):
	#对字典内容进行排序
	items=dictM.items()
	backitems=[[v[1],v[0]] for v in items]
	#backitems.sort(reverse=True)
	backitems.sort(reverse=False)
	return backitems

def buildmatrix(data,sup,datalength):
	#单项统计字典，便于根据支持度删值
	dictItem = {}
	for items in data:
		for item in items:
			if str(item) in dictItem.keys():
				dictItem[str(item)]+=1
			else:
				dictItem[str(item)]=1
	#以二维数组的方式存储字典,v[0]为计数，v[1]为items
	s1 = time.clock()
	#datalist_temp = dictquicksort(dictItem)
	datalist_temp1 = dictItem.items()
	datalist_temp = [[v[1],v[0]] for v in datalist_temp1]
	s2 = time.clock()
	print("===sorted===",(s2-s1))
	dllength = len(datalist_temp)
	datalist = []
	list_item = []
	list_item_d = {}
	for i in range(dllength):
		if datalist_temp[i][0]>=sup:
			list_item_d[datalist_temp[i][1]] = datalist_temp[i][0]
			list_item.append(datalist_temp[i][1])
	#获取到按照支持度排序后的L1项

	dictM = {}#去重字典
	#print("===len===list_item")
	#print(len(list_item))
	#list_dict_M = np.array(dictM.keys())
	#重复事件记录
	for i in range(len(data)):
		try:
			dictM[tuple(data[i])]+=1
		except KeyError:
			dictM[tuple(data[i])]=1
	#list_keys 出现过的事件
	print("===dictM====")
	print(dictM)
	list_keys = list(dictM.keys())
	list_ae = list(dictM.values())
	matrix = np.zeros((len(list_item),len(list_ae)))
	#print(len(list_item),len(list_ae))#109 8125
	#print(list_keys,len(list_keys[0]))#8125 23
	#构建满足频繁项集和数据长度的矩阵
	for i in range(len(list_keys)):
		for j in range(len(list_keys[i])):
			if list_keys[i][j] and list_keys[i][j] in list_item:
				#print('OK')
				n = list_item.index(list_keys[i][j])
				matrix[n][i]=1
	w = []
	for i in range(len(matrix)):
		temp = 0
		for j in range(len(matrix[0])):
			temp+=matrix[i][j]
		w.append(temp)
	return matrix,list_item,list_ae'''

'''def createL2(data, list_item, list_ae, sup):
	list_item_d = {}
	new_items = []
	new_matrix = []
	print("==in==create==L2===")
	length = len(list_item)
	for i in range(length):
		start = time.clock()
		for j in range(i + 1, length, 1):
			temp = []
			temp = np.multiply.reduce([data[i], data[j]])
			itemsup = np.dot(temp, list_ae)
			if itemsup >= sup:
				new_matrix.append(temp)
				new_items.append([list_item[i], list_item[j]])
				list_item_d[list_item[i], list_item[j]]=itemsup
	new_matrix = np.array(new_matrix)
	return new_matrix,list_item_d, new_items'''


def matrix(data, list_ae):
	temp_array1 = []
	for i in range(len(data[0])):
		temp_array = [j * list_ae[i] for j in data[:, i]]
		temp_array1.append(temp_array)
	temp_array1 = np.array(temp_array1)
	return temp_array1


def createL2_multimatrix(data, list_item, list_ae, sup):
	list_item_d = {}
	new_array_matrix = []
	new_list_item = []
	ma = matrix(data, list_ae)
	ma = np.array(ma)
	L2_matrix_temp = np.dot(ma.T, ma)
	L2_matrix = np.array(L2_matrix_temp)
	for i in range(L2_matrix.shape[0]):
		for j in range(i + 1, L2_matrix.shape[1], 1):
			if L2_matrix[i][j] >= sup:
				temp_array = []
				temp_array = data[i] * data[j]
				item_temp = tuple([list_item[i], list_item[j]])
				new_array_matrix.append(temp_array)
				new_list_item.append(item_temp)
				list_item_d[item_temp] = L2_matrix[i][j]

	return new_array_matrix,list_item_d, new_list_item


def aprioriGen(data, list_item, list_ae, sup, loop_num):
	list_item_d = {}
	print("loop_num",loop_num)
	new_items = []
	new_matrix = []
	for i in range(len(list_item)):
		start = time.clock()
		flag = 0
		for j in range(i + 1, len(list_item), 1):
			if flag == 1:
				break
			if list_item[i][:loop_num - 1] != list_item[j][:loop_num - 1]:
				flag = 1
				break
			else:
				temp = []
				new_temp = []
				temp = np.multiply.reduce([data[i], data[j]])
				itemsup = np.dot(temp, list_ae)
				if itemsup >= sup:
					new_matrix.append(temp)
					new_temp.extend(list_item[i])
					new_temp.append(list_item[j][-1])
					#print(new_temp)
					new_items.append(new_temp)
					list_item_d[tuple(new_temp)] = itemsup

		end = time.clock()
	length = len(new_matrix)
	if length >= 1:
		df = np.array(new_matrix)
		return df,list_item_d, new_items
	else:
		df = []
		new_items = []
		list_item_d={}
		return df,list_item_d, new_items

def cutsize_by_firstitem_count(df, new_items, loop_num):
	# 根据首项出现的次数进行剪枝，小于k
	dictM = {}
	for i in range(len(new_items)):
		try:
			dictM[new_items[i][0]] += 1
		except:
			dictM[new_items[i][0]] = 1
	items = dictM.items()
	newitems = [[v[0], v[1]] for v in items]
	newitems.sort()
	items2 = set()
	for i in range(len(newitems)):
		# print(newitems[i][1],type(newitems[i][1]))
		if int(newitems[i][1]) < loop_num:
			items2.add(newitems[i][0])
	newdf = []
	newitem1 = []
	for i in range(len(new_items)):
		if new_items[i][0] not in items2:
			newdf.append((df[i]))
			newitem1.append(new_items[i])

	df = np.array(newdf)
	return df, newitem1


def cutsize_base_t_length(data, loop_num, list_ae,list_length):
	# 根据事务长度来缩减规模
	list_length = np.array(list_length)
	list_ae = np.array(list_ae)
	data = np.array(data)
	return data[:,list_length>loop_num],list_ae[list_length>loop_num],list_length[list_length>loop_num]

def check(data, j):
	list_single = []
	list_local = []
	size = len(data)
	sli = []
	for i in range(size - 1, -1, -1):
		sli.append(data[i][j])
		if data[i][j] not in list_single:
			list_single.append(data[i][j])
	# print(list_single)
	# print(sli)
	for i in list_single:
		temp = sli.index(i)
		# print(temp)
		list_local.insert(0, size - 1 - temp)
	return list_local



def cutsize_by_item_count(data,new_items,loop_num):
	content = []
	new_items_set = []
	size = len(data)
	list_mark = np.ones((size))
	for i in range(3):
		l1 = check(data,i)
		l1.insert(0,-1)
		l2 = [[l1[m], l1[m + 1]] for m in range(len(l1) - 1)]
		for pair in l2:
			start = pair[0]
			dict1 = {}
			list_infreq = set()
			while start <= pair[1]:
				try:
					dict1[data[start][i]] += 1
				except:
					dict1[data[start][i]] = 1
				start += 1
			for key in dict1.keys():
				if dict1[key] < loop_num - i:
					list_infreq.add(key)
			for j in range(pair[0] + 1, pair[1] + 1):
				if data[j][i] in list_infreq:
					list_mark[j] = 0
	for i in range(size):
		if list_mark[i] == 1:
			content.append(data[i])
			new_items_set.append(new_items[i])#这部分可以用numpy的特性来改写提升效率，然而我这边各种报error
	return content,new_items_set




def main():
	frequent_itemsets = {}
	s1 = time.clock()
	name = "userid,Districtid,streetId,communityid,serviceid,money"
	data = dataimport(name)
	minsupport = 0.4
	datalength = len(data)
	sup = minsupport * datalength
	df, list_item_d,list_item, list_ae,list_length = buildmatrix(data,sup,datalength)
	frequent_itemsets.update(list_item_d)
	print("=======createL2============")
	#df,list_item_d,new_items = createL2(df,list_item,list_ae,sup)
	df,list_item_d, new_items = createL2_multimatrix(df, list_item, list_ae, sup)
	#print("L2",new_items)
	loop_num = 2
	frequent_itemsets.update(list_item_d)
	while len(df) > loop_num:
		print("=======createL"+str(loop_num+1)+"============")
		df,list_item_d, new_items = aprioriGen(df, new_items, list_ae, sup, loop_num)
		print(loop_num,new_items)
		frequent_itemsets.update(list_item_d)
		print(len(new_items))
		if len(df) < loop_num:#项集判定终止
			break
		if loop_num >= 3 and len(df)>loop_num:
			df, list_ae,list_length = cutsize_base_t_length(df, loop_num, list_ae,list_length) #根据事务长度进行压缩
			df, new_items = cutsize_by_firstitem_count(df, new_items, loop_num) #根据首项的数目进行压缩
			df,new_items = cutsize_by_item_count(df, new_items,loop_num)  #根据项的位置进行压缩矩阵
		loop_num += 1
		end = time.clock()

	s2 = time.clock()

	print("===Ending==")
	print(frequent_itemsets)
	print(sup)
	print((s2 - s1))


main()
