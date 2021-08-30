# study_pytorch
작성일자 : 2021년 8월 30일

- *참조* : Readme를 위해 사용할 글 작성 양식이나 세부적인 내용은 필요에 따라 계속 변경할 것임.

## 1. 목적
- Python 머신러닝 라이브러리인 Pytorch에 대한 공부

## 2. 사용 언어
- Language : Python
- Library : Pytorch

## 3. 디렉토리 구조

  engine 
  
    └__pycache__
    
    └__init__.py
    
    └learn.py
    
    └model.py
    
  resource
  
    └FashionMNIST
    
      └parameter
      
      └raw
      
  storage
  
  main.py
  
  ML.py
  
  README.md
  

## 4. 실행파일의 목적

  main.py : 1) 학습된 머신러닝 코드와 파라메터를 호출 
  
            2) 주어진 데이터에 대한 inference 수행
  
  ML.py : 머신러닝 학습을 수행
  
  engine/model.py : 머신러닝 구조를 제공함.
  
                    class NeuralNetwork  - 정상작동. 노드(784, 512, 512, 10)
                    
                    class CNN            - 현재 수정중
                    
  engine/learn.py : 머신러닝 학습을 위한 함수를 제공함.
  
                    function train       - 학습 기능 제공
                    
                    function test        - 테스트 기능 제공
                    

## 5. resource 폴더

  - 이 폴더에는 데이터셋을 위한 raw파일과 학습 결과인 parameter파일를 저장한다.

  - 디렉토리 템플릿

    [dataset] 
    
              - [parameter]
    
              - [raw]



## Referenece

https://pytorch.org/docs/stable/index.html


## Appendix

버전 관리 규칙

- 여기서 버전은 임시로 X.Y.Z라 정의한다.

  Z의 변경
    - 파일내의 수정 사항이 존재하는 경우.
    - 단, Readme의 수정사항은 제외한다.
    
  Y의 변경
    - 디렉토리 내의 추가된 파일이 존재하는 경우.
    - Y가 변경된 경우, Z는 1부터 다시 작성.
    
  X의 변경
    - 디렉토리 전체의 설계 구조가 변경된 경우.
    - X가 변경된 경우, Z는 1부터, Y는 0부터 다시 작성.
