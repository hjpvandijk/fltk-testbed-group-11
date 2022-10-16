g1 = {'2'; '2'; '2'; '2'; '2'; '2'; '2'; '2'; '2'; '500m'; '2'; '2'; '2'; '2'; '2'; '2'; '2'; '2'; '500m'; '500m'; '2'; '2'; '2'; '2'; '2'};
g2 = {'1Gi'; '1Gi'; '1Gi'; '2Gi'; '2Gi'; '1Gi'; '2Gi'; '2Gi'; '1Gi'; '1Gi'; '2Gi'; '2Gi'; '2Gi'; '2Gi'; '1Gi'; '2Gi'; '2Gi'; '2Gi'; '1Gi'; '2Gi'; '1Gi'; '1Gi'; '2Gi'; '2Gi'; '2Gi'};
g3 = [32 128 32 32 128 32 128 128 32 128 32 128 128 128 32 32 32 128 32 128 128 128 128 32 32];
g4 = [32 128 32 128 32 128 32 128 32 128 128 128 32 32 32 128 32 128 32 128 128 32 128 32 32];
g5 = [5 5 2 5 5 2 2 2 2 2 5 5 5 2 5 2 2 5 2 2 5 2 2 5 5];
g6 = {'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10CNN'; 'Cifar10ResNet'; 'Cifar10ResNet'};
y = [296638.0 238722.0 5921059.0 311043.0 228454.0 682937.0 6178311.0 492767.0 6819984.0 2236716.0 306002.0 2527669.0 234001.0 5344646.0 436158.0 4624282.0 682309.0 2579357.0 2685365.0 22166244.0 315698.0 382617.0 573797.0 2905844.0 2782493.0]';
p = anovan(y,{g1,g2,g3,g4,g5,g6},'varnames',{'cores', 'memory', 'train_batch', 'test_batch', 'parallel', 'network'},'alpha',0.1)