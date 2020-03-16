# MULTI GPU환경에서 ETRI 한국어 BERT모델 활용한 Korquad 학습 방법  

ETRI 에서 제공해주신 [한국어 BERT 모델](http://aiopen.etri.re.kr/)을 활용하여 한국어 기계독해(Korquad)를 **Multi GPU 환경**에서 **Tensorflow** 로 학습할 수 있도록 하였습니다.
(코드는 ETRI에서 제공해주신 pytorch 코드를 보고 변경하였습니다.)
형태소 분석기는 세종 품사 태그로 바꾼 [Mecab 형태소분석기 API](https://github.com/Gyunstorys/nlp-api) 를 적용하였습니다. 
Dependency 는 기존 run_squad_morph.py 와 같습니다. (python >=3.5 , tensorflow >=1.12 , urllib3)

**ETRI 의 [한국어 BERT 모델](http://aiopen.etri.re.kr/)을 활용하시려면 다음의 [링크](http://aiopen.etri.re.kr/service_dataset.php) 에서 서약서를 작성하시고 키를 받으셔서 다운받으시면 됩니다. 
(사용 허가 협약서를 준수하기 때문에 pretrained 모델을 공개하지 않습니다.**)

## 실행방법 


## Requirements
Python3.6

tensorflow==1.14.0 

urllib3

## 실행방법 

**step 1. [Mecab 형태소분석기](https://github.com/Gyunstorys/nlp-api) API 설치합니다.** 

**step 2. ETRI_pretrained directory 만들어서 ETRI pretrained model checkpoint 저장합니다.** 

**step 3. shell script path 바꿔줍니다. (docker 사용하면 mount 시킬 volume의 source path 만 바꿔주세요.)**
    BERT_BASE_DIR 를 clone 하신 path로 바꿔주세요 
    
**step 4. 실행합니다.**
    ./run_squad_ETRI.sh

## Korquad Result (dev set)

| model | Exact Match  |  F1 | 
| ------ | ------ | ------ | 
|ETRI+mecab| 84.08 |92.46 |


## 변경된 부분 

#### Tokenization 
Tokenizer는 ETRI 에서 제공해주신 `tokenization_morp.py`를 그대로 활용하였습니다. 
```python
import src_tokenizer.tokenization_morp as tokenization
```

##### run_squad.py 변경부분 
`class SquadExample`  를 ETRI의 `run_squad_morph.py` 에 있는 `class SquadExample` 로 대체했습니다. 

```python
def do_lang(text):
  openApiURL = "http://localhost:8080/api/morpheme/etri"

  http = urllib3.PoolManager()
  response = http.request("POST", openApiURL, fields={'targetText': text})

  return response.data.decode()
```

represent_ndoc 함수를 다음과 같이 바꿨습니다.
```python
def represent_ndoc(p_json):
  text = ''
  morp_list = []
  position_list = []
  for sent in p_json:

    text = sent['text']

    for morp in sent['morp']:
      morp_list.append(morp['lemma']+'/'+morp['type'])
      position_list.append(int(morp['position']))

  return {'text': text, 'morp_list': morp_list, 'position_list': position_list}

```
아래 함수는 형태소분석 API 가 return 한 Json 포맷에 맞췄습니다.

* `mapping_answer_korquad` 함수 안에 `p_json['sentence']` 를 `p_json`로  바꿔주세요.
(ETRI Json 포맷과 약간 다름니다. 유의해주세요.)
* `convert_examples_to_features` 
* `read_squad_examples` --> `read_squad_examples_and_do_lang` 

자세한건 코드를 참고해주세요.(너무 길어서...)



### Multi GPU 활용
Tensorflow 에서 Multi GPU 환경으로 학습하시려면 
* tf.contrib.distribute.CollectiveAllReduceStrategy
* tf.contrib.distribute.MirroredStrategy

과 같은 함수를 활용해야 합니다. 
[참고자료](https://www.youtube.com/watch?v=bRMGoPqsn20&t=381s)

#### run_squad.py 변경부분 

argument 로 다음을 추가해주시고 

```python
flags.DEFINE_integer("num_gpus", 8, "Total number of GPUs to use.")
flags.DEFINE_bool("multi_worker", False, "Multi-worker training.")
```

run_config 부분을 다음과 같이 바꿨습니다. 
```python
if FLAGS.multi_worker:
  distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
      num_gpus_per_worker=1)
  run_config = tf.estimator.RunConfig(
      experimental_distribute=tf.contrib.distribute.DistributeConfig(
          train_distribute=distribution,
          remote_cluster={
              'worker': ['localhost:5000', 'localhost:5001'],
          },
      )
  )
else:
  distribution = tf.contrib.distribute.MirroredStrategy(
      num_gpus=FLAGS.num_gpus)
  run_config = tf.estimator.RunConfig(train_distribute=distribution)
```

estimator 는 다음과 같이 바꿨습니다.  
```python
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={
        'batch_size': FLAGS.train_batch_size if FLAGS.do_train else FLAGS.predict_batch_size,
    }
)
```

#### optimization.py 변경 부분 
AdamWeightDecayOptimizer 가 multi-gpu 환경에서 동작하지 않기 때문에 
AdamOptimizer 로 바꿨습니다.
( 실험해보진 않았지만 TPU 를 활용할 때 와 성능 차이가 많이 나지는 않을것같습니다.)

```python
 optimizer = tf.train.AdamOptimizer(
 learning_rate=learning_rate,
 beta1=0.9,
 beta2=0.999,
 epsilon=1e-6
 )
```

### partner 
https://github.com/Gyunstorys
