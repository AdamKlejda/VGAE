class Weights():
    # new weight = capped(max/cur)
    min_weight = 1.0
    max_weight = 5.0
    weight_loss_A = 1.0
    weight_loss_X = 1.0
    weight_custom_loss = 1.0
    weight_mask_loss = 1.0

    # def __init__(self,min_weight=0.1,max_weight=10.0,weight_loss_A=1.0,weight_loss_X=1.0,weight_custom_loss=1.0,weight_mask_loss=1.0):
    #     self.min_weight = min_weight
    #     self.max_weight = max_weight
    #     self.weight_loss_A = weight_loss_A
    #     self.weight_loss_X = weight_loss_X
    #     self.weight_custom_loss = weight_custom_loss
    #     self.weight_mask_loss = weight_mask_loss
    
    def adjust_weight(self,weight):
        # takes weight as an imput and makes sure that the it is in the allowed range of min_weight and max_weight
        if weight > self.max_weight:
            return self.max_weight
        elif weight < self.min_weight:
            return self.min_weight
        else:
            return weight

    def set_weights_for_loss(self,losses,epoch):
        # loss, reconstruction_loss, reconstruction_lossA, reconstruction_lossX, reconstruction_lossMask = losses
        
        custom_loss = losses[0]-losses[1]
        new_max_loss = losses[0]/4 #2000-(10*epoch)/4
        self.weight_loss_A = self.adjust_weight(new_max_loss/losses[2])

        self.weight_loss_X = self.adjust_weight(new_max_loss/losses[3])
        
        if custom_loss > 0:
            self.weight_custom_loss = self.adjust_weight(new_max_loss/custom_loss)
        
        self.weight_mask_loss = self.adjust_weight(new_max_loss/losses[4])

    def print_weights_for_loss(self):
        print("weight_loss_A: ",self.weight_loss_A)
        print("weight_loss_X: ",self.weight_loss_X)
        print("weight_custom_loss: ",self.weight_custom_loss)
        print("weight_mask_loss: ",self.weight_mask_loss)