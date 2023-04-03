class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(DiceLoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
       
        N = inputs.size()[0]
       

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
        
        
        # Numerator Product
        target_oneHot = self.to_one_hot(target)

        inter = inputs * target_oneHot
        
        intersection = 0
        smooth = 1

        for imageInd in range(N): 
          intersection += inter[imageInd][1].view(-1).sum() 
          intersection += inter[imageInd][0].view(-1).sum()

        inter = 2*intersection + smooth
        #Denominator 
        union= inputs + target_oneHot
        ## Sum over all pixels N x C x H x W => N x C

        total2 = 0
        for imageInd in range(N):
          total2 += union[imageInd][1].view(-1).sum() 
          total2 += union[imageInd][0].view(-1).sum() 

        union = total2 + smooth
        

        loss = inter/union
        ## Return average loss over classes and batch
        return 1-loss.mean()