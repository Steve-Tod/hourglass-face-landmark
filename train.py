import argparse, random, pprint, os
from tqdm import tqdm
from options.options import parse_opt
from utils import utils



def main():
    parser = argparse.ArgumentParser(description='Train Hourglass Model')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = parse_opt(parser.parse_args().opt)
    
    from data import create_dataloader
    from solver.HourGlassSover import HourGlassSover

    pprint.pprint(opt)
    train_dl = create_dataloader(opt['datasets']['train'])
    print('===> Train Dataset created, Number of images: [%d]' % (len(train_dl) * opt['datasets']['train']['batch_size']))
    val_dl = create_dataloader(opt['datasets']['val'])
    print('===> Validation Dataset created, Number of images: [%d]' % (len(val_dl)))

    solver = HourGlassSover(opt)

    if opt['use_tb_logger']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger_root'])
        print('===> tensorboardX logger created, log to %s' % (opt['path']['tb_logger_root']))

    NUM_EPOCH = opt['train']['num_epochs']
    current_step = 0
    start_epoch = solver.cur_epoch
    print('===> Starting from epoch %d' % start_epoch)
    
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        solver.cur_epoch = epoch
        
        print('===> Train epoch %3d, learning rate: %.7f' % (epoch, solver.get_current_learning_rate()))
        solver.records['lr'].append(solver.get_current_learning_rate())
        with tqdm(total=len(train_dl), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            train_loss = []
            for i, sample in enumerate(train_dl):
                solver.feed_data(sample)
                loss = solver.train_step()
                train_loss.append(loss)
                if current_step % opt['train']['log_interval'] == 0 and current_step > 0:
                    tb_logger.add_scalar('train loss', loss, global_step=current_step)
                current_step += 1
                t.set_postfix_str("Batch Loss: %.4f" % loss)
                t.update()
        mean_train_loss = sum(train_loss) / len(train_loss)
        solver.records['train_loss'].append(mean_train_loss)
        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch, NUM_EPOCH, mean_train_loss))
        
        print('===> Validate epoch %3d' % (epoch))
        val_loss = []
        nme_all = []
        with tqdm(total=len(val_dl), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            for i, sample in enumerate(val_dl):
                solver.feed_data(sample)
                img_name = os.path.basename(sample['path'][0])
                loss = solver.evaluate()
                val_loss.append(loss)
                nme = solver.calc_nme()
                nme_all.append(nme)
                if i < opt['train']['num_save_image']:
                    solver.log_current_visual(img_name, tb_logger, current_step)
                    solver.save_current_visual(img_name, epoch)
                t.set_postfix_str("Val Loss: %.4f, NME: %.4f" % (loss, nme))
                t.update()
              
        epoch_is_best = False
        mean_val_loss = sum(val_loss) / len(val_loss)
        mean_nme = sum(nme_all) / len(nme_all)
        if solver.best_pred > mean_nme:
            solver.best_pred = mean_nme
            solver.best_epoch = epoch
            epoch_is_best = True
            
        solver.records['val_loss'].append(mean_val_loss)
        solver.records['nme'].append(mean_nme)
        if opt['use_tb_logger']:
            tb_logger.add_scalar('val loss', mean_val_loss, global_step=current_step)
            tb_logger.add_scalar('nme', mean_nme, global_step=current_step)
        
        print('Epoch: %03d, Mean NME: %.4f, Mean loss: %.4f\nBest Epoch: %03d, Best NME: %.4f' % 
             (epoch, mean_nme, mean_val_loss, solver.best_epoch, solver.best_pred))
        
        solver.update_learning_rate(epoch)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        
    print('===> Finished !')
        
if __name__ == '__main__':
    main()
