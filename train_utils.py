def save_weights(model, epoch, loss, err, WEIGHTS_PATH):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.pth')
 
def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch
 
def save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH, plot=True):
    loss_fname = 'losses-' + loss_label + '.pth'
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    torch.save(losses, loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
        
def load_loss(loss_fname, LOSSES_PATH, RESULTS_PATH, plot=True):
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    losses = torch.load(loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
 
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):    
        nn.init.xavier_normal(m.weight)
        m.bias.data.zero_()
 
def plot_loss_lcc(losses, RESULTS_PATH):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    trn_ed, val_ed = [], []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        trn_ed.append(losses[i][2])
        val_loss.append(losses[i][3])
        val_ed.append(losses[i][4])
    #plot loss
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Loss curves')
    ax1.plot(epochs, trn_loss, label='train loss')
    ax1.plot(epochs, val_loss, label='val loss')
    ax1.plot(epochs, trn_ed, '-r', label='train lcc')
    ax1.plot(epochs, val_ed, '-b', label='val lcc')
    #set legend in the fig
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'newlossfig.png')
    plt.show()
 
import torch
#from torch.autograd import Variable
#import torch.nn.functional as F
    
def train(model, trn_loader, optimizer, criterion, epoch, mr):
    model.train()
    trn_loss, trn_lcc, trn_mae = 0, 0, 0 
    for idx, (mov, ref) in enumerate(trn_loader): 
        mov = mov.cuda()
        ref = ref.cuda()
 
        optimizer.zero_grad()
        if mr:
            warped0, warped, ref, flow = model(mov, ref)
            loss0 = criterion['lcc'](warped0, ref)
            loss1 = criterion['lcc'](warped, ref)
            loss2 = criterion['grad'](flow)
            loss = criterion['gamma'] * loss0 + loss1 + criterion['lambda'] * loss2
        else:
            warped, ref, flow = model(mov, ref) #, mask 
            loss1 = criterion['lcc'](warped, ref)
            loss2 = criterion['grad'](flow)
            loss = loss1 + criterion['lambda'] * loss2 + criterion['gamma'] * flow.abs().mean()
        
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        trn_lcc += loss1.item()
        trn_mae += torch.mean(torch.abs(flow)).item()
 
    trn_loss /= len(trn_loader) 
    trn_lcc /= len(trn_loader) 
    trn_mae /= len(trn_loader) 
    return trn_loss, trn_lcc, trn_mae
 
def test(model, test_loader, criterion, epoch, mr):
    model.eval()
    test_loss, test_lcc, test_mae = 0, 0, 0
    for mov, ref in test_loader:
        mov = mov.cuda()
        ref = ref.cuda()
        with torch.no_grad():
            if mr:
                warped0, warped, ref, flow = model(mov, ref)
                loss0 = criterion['lcc'](warped0, ref)
                loss1 = criterion['lcc'](warped, ref)
                loss2 = criterion['grad'](flow)
                loss = criterion['gamma'] * loss0 + loss1 + criterion['lambda'] * loss2
            else:
                warped, ref, flow = model(mov, ref) #, mask 
                loss1 = criterion['lcc'](warped, ref)
                loss2 = criterion['grad'](flow)
                loss = loss1 + criterion['lambda'] * loss2 + criterion['gamma'] * flow.abs().mean()
                
        test_loss += loss.item()
        test_lcc += loss1.item()
        test_mae += torch.mean(torch.abs(flow)).item()
    
    test_loss /= len(test_loader) 
    test_lcc /= len(test_loader) 
    test_mae /= len(test_loader) 
    return test_loss, test_lcc, test_mae