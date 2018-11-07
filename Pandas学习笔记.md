
# Pandas的数据类型

>- Series(一维数据结构)
>- Dataframe

#### Series --- 带标签的一维数组
常用的初始化方法：
- 可迭代对象
- np数组
- 字典对象
- 标量

# 一、Series

## 1. Series初始化

#### 导入


```python
import pandas as pd
import numpy as np
```


```python
s = pd.Series([1, 2, 3])
```


```python
type(s)
```




    pandas.core.series.Series




```python
s
```




    0    1
    1    2
    2    3
    dtype: int64



#### 通过可迭代对象创建Series


```python
 pd.Series(range(5))
```




    0    0
    1    1
    2    2
    3    3
    4    4
    dtype: int64



#### 通过numpy数组创建Series


```python
t = np.random.randint(5, 15, size=(8))
pd.Series(t)
```




    0    11
    1     9
    2     6
    3     7
    4    11
    5    12
    6     6
    7    14
    dtype: int64



#### 通过标量创建


```python
pd.Series(100, index=['a', 5, b'sd'])
```




    a        100
    5        100
    b'sd'    100
    dtype: int64



#### 使用字典自带索引


```python
pd.Series({100:165, 'asdf':961})
```




    100     165
    asdf    961
    dtype: int64



## 2. Series数据属性

## 2.1 索引

#### 获得索引


```python
s = pd.Series([7,8,9], index=[1,2,3])
s.index
```




    Int64Index([1, 2, 3], dtype='int64')




```python
s.index = ['a', 'b', 'c']
```


```python
s
```




    a    7
    b    8
    c    9
    dtype: int64



##### 可以手动创建Index对象（数量必须匹配）


```python
index = pd.Index(['aaaa', 'bbbb', 'cccc'])
pd.Series([7,8,9], index=index)
```




    aaaa    7
    bbbb    8
    cccc    9
    dtype: int64



## 2.2 值

#### 返回数据


```python
s.values
```




    array([7, 8, 9])




```python
s
```




    a    7
    b    8
    c    9
    dtype: int64



## 2.3 尺寸


```python
s.size
```




    3




```python
s.dtype
```




    dtype('int64')



## 2.4 其他

#### Series可以指定name


```python
index = pd.Index(['a', 'b', 'c'], name = 'Index名字')
```


```python
s = pd.Series([1,2,3], index=[1,2,3], name='"Series名字"')
s
```




    1    1
    2    2
    3    3
    Name: "Series名字", dtype: int64



#### 索引可以指定name属性


```python
s.index = index
s
```




    My_Index
    a    1
    b    2
    c    3
    Name: "Series名字", dtype: int64



#### head 和 tail ,默认(n=5)


```python
s.head(2)
```




    1    1
    2    2
    Name: "Series名字", dtype: int64




```python
s.tail(100)
```




    1    1
    2    2
    3    3
    Name: "Series名字", dtype: int64




```python
test_np = np.random.randint(0, 15, size = 10)
test_np
```




    array([3, 5, 9, 6, 1, 8, 9, 9, 2, 1])




```python
test_pd = pd.Series(test_np)
test_pd
```




    0     6
    1    11
    2     4
    3     3
    4     4
    5     9
    6     4
    7     7
    8    11
    9     5
    dtype: int64




```python
test_np[5]
```




    9




```python
test_pd[5]
```




    9




```python
test_np[5] == test_pd[5]
```




    True



## 3. Series运算


```python
test_pd
```




    0     7
    1    12
    2     5
    3     4
    4     5
    5    10
    6     5
    7     8
    8    12
    9     6
    dtype: int64




```python
test_pd + 1
```




    0     8
    1    13
    2     6
    3     5
    4     6
    5    11
    6     6
    7     9
    8    13
    9     7
    dtype: int64




```python
test_pd + test_pd
```




    0    14
    1    24
    2    10
    3     8
    4    10
    5    20
    6    10
    7    16
    8    24
    9    12
    dtype: int64



#### Series按照 **index** 计算，缺失则返回结果NaN（not a number）


```python
s1 = pd.Series([1,2,3], index=[1,2,3])
s2 = pd.Series([1,2,3], index=[2,3,4])
s1 + s2
```




    1    NaN
    2    3.0
    3    5.0
    4    NaN
    dtype: float64



#### 使用函数方式运算，如果需要处理不匹配值，那么对Series对象填充索引，指定填充值，并进行运算


```python
s1.add(s2, fill_value=100000)
```




    1    100001.0
    2         3.0
    3         5.0
    4    100003.0
    dtype: float64



#### 几个特殊浮点数, 以及空值的判断


```python
s = pd.Series([1, 2, 3, float('NaN'), np.NaN])
```


```python
s.isnull()
```




    0    False
    1    False
    2    False
    3     True
    4     True
    dtype: bool



#### nd 和 pd 在计算时对空值的的处理不同
* numpy会产生空值（）
* pandas忽略空值


```python
t = np.array([1, 2, 3, float('NaN'), np.NaN])
t.sum()
```




    nan




```python
s.sum()
```




    6.0



### 4. 提取元素

- 通过***索引***提取元素
- 通过***标签数组***和***布尔数组***提取元素(推荐)


```python
a = np.array([1, 2, 3])
b = pd.Series(a, index = [0,1,2])
```


```python
index1 = [0, 1, 2]
index2 = [False, True, True]
```


```python
b[index1]
```




    0    1
    1    2
    2    3
    dtype: int64




```python
b[index2]
```




    1    2
    2    3
    dtype: int64



### 注意：
 - 访问可以使用标签索引，也可以使用位置索引
 - 创建时指定的标签，称为标签索引，如果标签索引是数值类型，替换原先默认的位置索引（位置索引失效）

### 5. 标签索引（loc）和位置索引（iloc） --- 避免索引混淆


```python
b.loc[0]
```




    1




```python
b.iloc[1]
```




    2




```python
# test_np = np.random.randint(0, 15, size = 10)
test_np = np.arange(15)
test_pd = pd.Series(test_np)

test_pd.loc[4:8].values # 标签索引会前闭后闭
```




    array([4, 5, 6, 7, 8])




```python
test_pd.iloc[4:8].values # 标签索引会前闭后开
```




    array([4, 5, 6, 7])




```python
test_np[4:8] # np索引前闭后开
```




    array([4, 5, 6, 7])




```python
test_np[4:80]
```




    array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])



### 6. 数值操作
- 获取值
- 修改值
- 增加索引-值
- 删除索引-值


```python
s = pd.Series([1, 2, 3, float('NaN'), np.NaN])
s.loc['a'] = 'a'
s
```




    0      1
    1      2
    2      3
    3    NaN
    4    NaN
    a      a
    dtype: object




```python
s.drop('a') # 创建新的删除对象
```




    0      1
    1      2
    2      3
    3    NaN
    4    NaN
    dtype: object




```python
s
```




    0      1
    1      2
    2      3
    3    NaN
    4    NaN
    a      a
    dtype: object




```python
s.drop(['a', 3], inplace=True) # 可以这样子删除,所有的inplace参数都默认为False,即返回新对象
s
```




    0      1
    1      2
    2      3
    4    NaN
    dtype: object



## 9. 其他
- unique        --- 去重，但是不排序
- value_counts  --- 计数


```python
s = pd.Series([1, 10, -2, -5, 20, 10, -5])
s.unique()
```




    array([ 1, 10, -2, -5, 20])




```python
s.value_counts(ascending=True)
```




     1     1
    -2     1
     20    1
     10    2
    -5     2
    dtype: int64





# 二、DataFrame类型



## 1. DataFrame创建
多维数据类型，常用在二维情况，包含行标签和列标签。二维DaraFrame的创建方式如下：
- 二维数组结构（列表，ndarray,DataFrame等）
- 字典类型，key为列标签，value为一维数据结构


```python
df1 = pd.DataFrame([[11, 21, 31], [99, 88, 77]])
df2 = pd.DataFrame([[11, 21, 31, 41], [99, 88, 77, 66]])
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>21</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>88</td>
      <td>77</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df1)
```

        0   1   2
    0  11  21  31
    1  99  88  77


#### IPython的扩展内建函数display() 可以把多个数据美化呈现方式


```python
display(df1)
display(df2)
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>21</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>88</td>
      <td>77</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>21</td>
      <td>31</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>88</td>
      <td>77</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>


#### DataFrame使用的是列向量，因此通过字典创建，Key是列表签名


```python
di = {
    "名字":['a', 'b', 'c', 'd'],
    '年龄':[32, 23, 45, 76],
    '班级':8,
    '成绩':np.random.randint(0,100,4)
}
df = pd.DataFrame(di)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>46</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>95</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>67</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### index是行标签， columns是列标签


```python
# df.index = ['张三', '李四'， '王五', '李六']
df.columns = ["学生名字", '学生年龄', '学生成绩', '学生班级']
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>学生名字</th>
      <th>学生年龄</th>
      <th>学生成绩</th>
      <th>学生班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>46</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>95</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>67</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



## 2. 抽取数据——抽样

#### 从头和尾取数据


```python
df.head(n=2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(n=2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### 随机取样


```python
df.sample(n=2, frac=None, replace=False, weights=None, random_state=None, axis=None) # 默认不放回抽样，抽取1个
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=10, replace=True, random_state=456) # random_state是随机数种子； replace=True 是放回抽样
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=10, replace=True, random_state=456) # random_state是随机数种子； replace=True 是放回抽样
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



## 3. DataFrame属性  
- index   --- 行索引
- columns --- 列索引
- values  --- 数据，二维ndarray数据
- shape   --- 形状
- ndim    --- 维数
- dtypes  --- 数据类型(ndarray是一个能存储的所有元素的唯一类型，DataFrame每一列一个类型)


```python
df.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
df.columns
```




    Index(['名字', '年龄', '成绩', '班级'], dtype='object')




```python
df.values
```




    array([['a', 32, 51, 8],
           ['b', 23, 53, 8],
           ['c', 45, 63, 8],
           ['d', 76, 79, 8]], dtype=object)




```python
df.shape
```




    (4, 4)




```python
df.ndim
```




    2



#### 返回的数据为Series类型


```python
df.dtypes
```




    名字    object
    年龄     int64
    成绩     int64
    班级     int64
    dtype: object



## 4. 行、列操作：  
- 可以通过index和columns提出特定数据
- 可以为index和columns创建name
- 直接中括号索引获取列，loc和iloc获取行（Series）

## 4.1 获取行、列


```python
di = {
    "名字":['a', 'b', 'c', 'd'],
    '年龄':[32, 23, 45, 76],
    '班级':8,
    '成绩':np.random.randint(0,100,4)
}
df = pd.DataFrame(di)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>68</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>33</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>35</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### 获取一个数据


```python
df.loc[0, '名字']
```




    'a'



#### 获取一行


```python
df[ '名字']
```




    0    a
    1    b
    2    c
    3    d
    Name: 名字, dtype: object



#### df [ ] 访问多列


```python
df[['名字', '年龄']]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>



#### 获取一行，每一行是一个Series类型


```python
df.loc[1]
```




    名字     b
    年龄    23
    成绩    33
    班级     8
    Name: 1, dtype: object



#### 访问多行


```python
df.loc[[1,2,3]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>33</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>35</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



## 4.2 增加行、列


#### 4.2.1 获取某一列，Series的索引name为DataFrame列标签名字


```python
df['@'] = [1,2,3,4]
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
      <th>@</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>68</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>33</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>35</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>11</td>
      <td>8</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.2.2 添加一列


```python
df['@']
```




    0    1
    1    2
    2    3
    3    4
    Name: @, dtype: int64




```python
df.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
df.index.name = '行索引名'
df.columns.name = '列索引名'
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>列索引名</th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
      <th>@</th>
    </tr>
    <tr>
      <th>行索引名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>68</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>33</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>35</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>11</td>
      <td>8</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



#### 添加一个求和列


```python
df1 = pd.DataFrame({
    '苹果':[1,2,3],
    '香蕉':[4,5,6],
    '葡萄':[7,8,9],
})
df1['总和'] = df1['苹果'] + df1['香蕉'] + df1['葡萄']
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
      <th>香蕉</th>
      <th>总和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.2.3 删除列


```python
df.pop('@')
```




    行索引名
    0    1
    1    2
    2    3
    3    4
    Name: @, dtype: int64



#### 4.2.4 获取行


```python
df.drop([1,2], axis='index', inplace=False) #返回新对象，不inplace修改
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>列索引名</th>
      <th>学生名字</th>
      <th>学生年龄</th>
      <th>学生成绩</th>
      <th>学生班级</th>
    </tr>
    <tr>
      <th>行索引名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>31</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>67</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[[2, 3]] # 推荐使用标签名称获取对象
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>列索引名</th>
      <th>学生名字</th>
      <th>学生年龄</th>
      <th>学生成绩</th>
      <th>学生班级</th>
    </tr>
    <tr>
      <th>行索引名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>95</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>67</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[2, 3]] # 不推荐使用
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>列索引名</th>
      <th>学生名字</th>
      <th>学生年龄</th>
      <th>学生成绩</th>
      <th>学生班级</th>
    </tr>
    <tr>
      <th>行索引名</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>95</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>67</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

#### 4.2.5 增加一行,需要添加的Series数据必须含有name(对应行标签)


```python
di = {
    "名字":['a', 'b', 'c', 'd'],
    '年龄':[32, 23, 45, 76],
    '班级':8,
    '成绩':np.random.randint(0,100,4)
}
df = pd.DataFrame(di)
row = pd.Series([ 's', 45, 65, 8], name='new', index=['名字', '年龄', '成绩', '班级'])
df.append(row )
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>87</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>74</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>36</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>13</td>
      <td>8</td>
    </tr>
    <tr>
      <th>new</th>
      <td>s</td>
      <td>45</td>
      <td>65</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
di = {
    "名字":['a', 'b', 'c', 'd'],
    '年龄':[32, 23, 45, 76],
    '班级':8,
    '成绩':np.random.randint(0,100,4)
}
dff = pd.DataFrame(di)
row = pd.Series([ 's', 45, 65, 8], name='new', index=['名字', '年龄', '成绩', '班级'])
dff.append(row, ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>93</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>73</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s</td>
      <td>45</td>
      <td>65</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### 在增加多行的时候，优先使用concat,性能更好。


```python
pd.concat((df, dff), axis=0, ignore_index=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>93</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>73</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.concat((df, dff), axis=1, ignore_index=False)
result
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
      <th>名字</th>
      <th>年龄</th>
      <th>成绩</th>
      <th>班级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>32</td>
      <td>51</td>
      <td>8</td>
      <td>a</td>
      <td>32</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>23</td>
      <td>53</td>
      <td>8</td>
      <td>b</td>
      <td>23</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>45</td>
      <td>63</td>
      <td>8</td>
      <td>c</td>
      <td>45</td>
      <td>93</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>76</td>
      <td>79</td>
      <td>8</td>
      <td>d</td>
      <td>76</td>
      <td>73</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.2.6 删除l列、行


```python
result.drop(['名字', '班级'], axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>成绩</th>
      <th>年龄</th>
      <th>成绩</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>51</td>
      <td>32</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23</td>
      <td>53</td>
      <td>23</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>63</td>
      <td>45</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>76</td>
      <td>79</td>
      <td>76</td>
      <td>73</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.2.7 混合操作
可以先获取行，也可以现货区列
- drop方法可以删除行和列；
- df[索引]针对列操作,不支持位置索引，只支持列标签；
- df.loc[索引]、df.iloc[索引]针对行操作；
- df[切片]不推荐【对行操作，既支持位置索引，也支持标签索引; 此外，和第二条冲突，切片索引变成了行操作不利于记忆】
- df[[列表]] 也存在歧义，如果是【标签数组- 列操作】【布尔数组- 行操作】


```python
df = pd.DataFrame({
    '苹果':[1,2,3],
    '香蕉':[4,5,6],
    '葡萄':[7,8,9],
})
df['总和'] = df['苹果'] + df['香蕉'] + df['葡萄']
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
      <th>香蕉</th>
      <th>总和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['苹果'].loc([0])
```




    <pandas.core.indexing._LocIndexer at 0x7fd51e4b37f0>




```python
df[['苹果', '葡萄']].loc[[0,2]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



#### 切片访问行


```python
df.iloc[0:1]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
      <th>香蕉</th>
      <th>总和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0]
```




    苹果     1
    葡萄     7
    香蕉     4
    总和    12
    Name: 0, dtype: int64



### 4.2.8 标签名，name属性的转换

- 如果拿出列数据，
- 如果拿出行数据，


```python
df = pd.DataFrame({
    '苹果':[1,2,3],
    '香蕉':[4,5,6],
    '葡萄':[7,8,9],
})
df['总和'] = df['苹果'] + df['香蕉'] + df['葡萄']
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
      <th>香蕉</th>
      <th>总和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>5</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[0]
```




    苹果     1
    葡萄     7
    香蕉     4
    总和    12
    Name: 0, dtype: int64




```python
df[[True, False, False]]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>苹果</th>
      <th>葡萄</th>
      <th>香蕉</th>
      <th>总和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['苹果']
```




    0    1
    1    2
    2    3
    Name: 苹果, dtype: int64



## 5. 计算


```python
df1 = pd.DataFrame(np.arange(24).reshape(4,6))
df2 = pd.DataFrame(np.arange(100, 124).reshape(4,6))
```

#### 转置


```python
df1.T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>12</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7</td>
      <td>13</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>14</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>9</td>
      <td>15</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>10</td>
      <td>16</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>11</td>
      <td>17</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



#### 加法


```python
df1 + df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>102</td>
      <td>104</td>
      <td>106</td>
      <td>108</td>
      <td>110</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112</td>
      <td>114</td>
      <td>116</td>
      <td>118</td>
      <td>120</td>
      <td>122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>124</td>
      <td>126</td>
      <td>128</td>
      <td>130</td>
      <td>132</td>
      <td>134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>136</td>
      <td>138</td>
      <td>140</td>
      <td>142</td>
      <td>144</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>



#### 加法对不齐，产生NaN


```python
df2.index = [0, 1, 3, 4]
df2.columns = [0, 1, 2, 3, 4, 6]
df1 + df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>102.0</td>
      <td>104.0</td>
      <td>106.0</td>
      <td>108.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112.0</td>
      <td>114.0</td>
      <td>116.0</td>
      <td>118.0</td>
      <td>120.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130.0</td>
      <td>132.0</td>
      <td>134.0</td>
      <td>136.0</td>
      <td>138.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.add(df2, fill_value=0)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>102.0</td>
      <td>104.0</td>
      <td>106.0</td>
      <td>108.0</td>
      <td>5.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112.0</td>
      <td>114.0</td>
      <td>116.0</td>
      <td>118.0</td>
      <td>120.0</td>
      <td>11.0</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130.0</td>
      <td>132.0</td>
      <td>134.0</td>
      <td>136.0</td>
      <td>138.0</td>
      <td>23.0</td>
      <td>117.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>118.0</td>
      <td>119.0</td>
      <td>120.0</td>
      <td>121.0</td>
      <td>122.0</td>
      <td>NaN</td>
      <td>123.0</td>
    </tr>
  </tbody>
</table>
</div>



#### DaraFram 和 Series 加法
   --- 行和列操作都可以操作


```python
s = pd.Series([100, 200, 300, 400, 500], index = np.arange(5))
```


```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



#### 默认列对齐


```python
df1 + s
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>201.0</td>
      <td>302.0</td>
      <td>403.0</td>
      <td>504.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106.0</td>
      <td>207.0</td>
      <td>308.0</td>
      <td>409.0</td>
      <td>510.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>112.0</td>
      <td>213.0</td>
      <td>314.0</td>
      <td>415.0</td>
      <td>516.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>118.0</td>
      <td>219.0</td>
      <td>320.0</td>
      <td>421.0</td>
      <td>522.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 也可以行操作


```python
df1.add(s,  axis='index')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>101.0</td>
      <td>102.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>206.0</td>
      <td>207.0</td>
      <td>208.0</td>
      <td>209.0</td>
      <td>210.0</td>
      <td>211.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>312.0</td>
      <td>313.0</td>
      <td>314.0</td>
      <td>315.0</td>
      <td>316.0</td>
      <td>317.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>418.0</td>
      <td>419.0</td>
      <td>420.0</td>
      <td>421.0</td>
      <td>422.0</td>
      <td>423.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 7. 排序
-  索引排序
-  值排序


```python
df = pd.DataFrame(np.arange(24).reshape(4,6), index=[5, 6, 2, 4], columns=[6,1,7,3, 4,2])
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>6</th>
      <th>1</th>
      <th>7</th>
      <th>3</th>
      <th>4</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(axis=1, ascending=False) # 列操作，降序操作
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>7</th>
      <th>6</th>
      <th>4</th>
      <th>3</th>
      <th>2</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>6</td>
      <td>10</td>
      <td>9</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>12</td>
      <td>16</td>
      <td>15</td>
      <td>17</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>18</td>
      <td>22</td>
      <td>21</td>
      <td>23</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(axis=0, ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>6</th>
      <th>1</th>
      <th>7</th>
      <th>3</th>
      <th>4</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(5, axis=1, ascending=False, inplace=False) # 行操作，降序操作 ************易混淆*****************
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>4</th>
      <th>3</th>
      <th>7</th>
      <th>1</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11</td>
      <td>10</td>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17</td>
      <td>16</td>
      <td>15</td>
      <td>14</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>22</td>
      <td>21</td>
      <td>20</td>
      <td>19</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



## 8. 统计方法
- mean / sum / count / median
- max / min
- cumsum / cumprod
- argmax / argmin (所在索引, 老式不推荐)
- idxmax / idxmin (所在索引，推荐)
- var / std (标准差， 方差)
- corr / cov (相关系数， 协方差)


```python
df = pd.DataFrame(np.arange(24).reshape(4,6))
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean(axis='columns')
```




    0     2.5
    1     8.5
    2    14.5
    3    20.5
    dtype: float64




```python
df.idxmax()
```




    0    3
    1    3
    2    3
    3    3
    4    3
    5    3
    dtype: int64




```python
df.var
```




    <bound method DataFrame.var of     0   1   2   3   4   5
    0   0   1   2   3   4   5
    1   6   7   8   9  10  11
    2  12  13  14  15  16  17
    3  18  19  20  21  22  23>




```python
 df.std
```




    <bound method DataFrame.std of     0   1   2   3   4   5
    0   0   1   2   3   4   5
    1   6   7   8   9  10  11
    2  12  13  14  15  16  17
    3  18  19  20  21  22  23>




```python
df.corr()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



> cov(X,Y) = E( [X - E(X)] [Y - E(Y)] )


```python
df.cov()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div>


