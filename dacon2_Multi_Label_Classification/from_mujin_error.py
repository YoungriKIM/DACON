# 무진님이 주신 에러 대응 코드

# 실행 하는 곳이 메인인 경우
if __name__ == '__main__':
    
    # 옵티마이저와 멀티라벨소프트 마진 로스를 사용함
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MultiLabelSoftMarginLoss()

    # 에포치 10주고 모델을 트레인으로 변환
    num_epochs = 40
    model.train()

    # 에포치 만큼 반복
    for epoch in range(num_epochs):
            # 배치 사이즈 만큼 스탭을 진행함
            for i, (images, targets) in enumerate(train_loader):

                # 미분 값 초기화
                optimizer.zero_grad()
                # 데이터셋을 프로세스에 입력함
                images = images.to(device)
                targets = targets.to(device)
                # 모델에 인풋을 넣고 아웃풋을 출력함
                outputs = model(images)
                # 로스를 확인함
                loss = criterion(outputs, targets)

                # 로스 역전파
                loss.backward()
                # 매개변수 갱신함
                optimizer.step()
            
                # 10에포치 마다 로스와 액큐러시를 출력함
                if (i+1) % 10 == 0:
                    outputs = outputs > 0.5
                    acc = (outputs == targets).float().mean()
                    print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')

if __name__ == '__main__':

    # 평가 폴더를 열음
    submit = pd.read_csv('data/sample_submission.csv')

    # 이벨류 모드로 전환
    model.eval()

    # 베치사이즈는 테스트로더 베치사이즈
    batch_size = test_loader.batch_size
    # 인덱스 0부터 시작
    batch_index = 0
    # 이벨류 모드를 테스트 셋으로 진행하고 파일에 입력함
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        outputs = outputs > 0.5
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    # 저장함
    submit.to_csv('submit.csv', index=False)