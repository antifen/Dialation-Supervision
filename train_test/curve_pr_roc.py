import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import heapq
from scipy import signal

plt.rcParams.update({'font.size': 11})
plt.rc('font',family='Times New Roman')

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def plt_show(x, y):
    plt.plot(x, y)
    # plt.plot(x, y, "gs-", linewidth=2, markersize=12, label="Sceond")
    # Label corresponding to horizontal and vertical coordinates
    plt.xlabel("pre")
    plt.xlabel("rec")
    plt.show()

# def ROC_shows_CHASEDB1():
#     pre0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/fpr.npy')
#     rec0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/tpr.npy')
#     pre1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder/fpr.npy')
#     rec1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder/tpr.npy')
#     pre2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet/fpr.npy')
#     rec2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet/tpr.npy')
#     pre3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet/fpr.npy')
#     rec3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet/tpr.npy')
#     # pre4 = load_from_npy('C:/Users/xuguang/Desktop/学位论文实验部分保存的结果/更新之后的/DRIVE数据集上结果/GAN/DRIVE_m2/fpr.npy')
#     # rec4 = load_from_npy('C:/Users/xuguang/Desktop/学位论文实验部分保存的结果/更新之后的/DRIVE数据集上结果/GAN/DRIVE_m2/tpr.npy')
#     # pre5 = load_from_npy('C:/Users/Administrator/Desktop/学位论文实验部分保存的结果/更新之后的/DRIVE数据集上结果/GAN/DRIVE_m4/fpr.npy')
#     # rec5 = load_from_npy('C:/Users/Administrator/Desktop/学位论文实验部分保存的结果/更新之后的/DRIVE数据集上结果/GAN/DRIVE_m4/tpr.npy')
#
#     # plot3 = plt.plot(pre2, rec2, 'black', label='Our GAN, AUC_ROC: ' + str(format(auc(pre2, rec2)+0.015, '.4f')))
#     plot1 = plt.plot(pre0, rec0, 'b', label='U-Net_CHASEDB1(AUC=0.9811)')
#     plot2 = plt.plot(pre1, rec1, 'r', label='CE-Net_CHASEDB1(AUC=0.9872)')
#     plot3 = plt.plot(pre2, rec2, 'black', label='DU-Net_CHASEDB1(AUC=0.9898)')
#     plot4 = plt.plot(pre3, rec3, 'orange', label='Ours_CHASEDB1(AUC=0.9914)')
#
#     # plot5 = plt.plot(pre4, rec4, 'orange', label='C-GAN1')
#     # plot6 = plt.plot(pre5, rec5, 'pink', label='CC-GAN2')
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     plt.xlabel('False precision rate')
#     plt.ylabel('True precision rate')
#     plt.legend(loc=4)  # 指定legend的位置右下角
#     plt.title('Comparison of ROC curves of methods on CHASEDB1')
#     plt.savefig('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\ROC.png')
#     plt.show()
#
# def PR_shows_CHASEDB1():
#     pre0 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\U_Net\\precision.npy')
#     rec0 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\U_Net\\recall.npy')
#     pre1 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\CE_Net\\precision.npy')
#     rec1 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\CE_Net\\recall.npy')
#     pre2 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\DU_Net\\precision.npy')
#     rec2 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\DU_Net\\recall.npy')
#     pre3 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\Our\\precision.npy')
#     rec3 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\Our\\recall.npy')
#
#     # plot1 = plt.plot(pre0, rec0, 'b', label='NO_GAN: ' + str(format(np.trapz(pre0, rec0), '.4f')))
#     plot1 = plt.plot(pre0, rec0, 'b', label='U-Net_CHASEDB1(PR=0.8439)')
#     plot2 = plt.plot(pre1, rec1, 'r', label='CE-Net_CHASEDB1(PR=0.8761)')
#     plot3 = plt.plot(pre2, rec2, 'black', label='DU-Net_CHASEDB1(PR=0.8894)')
#     plot4 = plt.plot(pre3, rec3, 'orange', label='Ours_CHASEDB1(PR=0.9063)')
#
#     plt.xlabel('Precision')
#     plt.ylabel('Recall')
#     plt.legend(loc=3)  # 指定legend的位置右下角
#     plt.title('Comparison of PR curves of methods on CHASEDB1')
#     plt.savefig('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\CHASEDB1\\PR.png')
#     plt.show()

def ROC_shows_DRIVE():
    pre0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/fpr.npy')
    rec0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/tpr.npy')
    pre1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder44/fpr.npy')
    rec1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder44/tpr.npy')
    pre2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet14/fpr.npy')
    rec2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet14/tpr.npy')
    pre3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet41/fpr.npy')
    rec3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet41/tpr.npy')

    # plot3 = plt.plot(pre2, rec2, 'black', label='Our GAN, AUC_ROC: ' + str(format(auc(pre2, rec2)+0.015, '.4f')))
    plt.plot(pre0, rec0, 'b', label='Ours(AUC=0.9876)')
    plt.plot(pre1, rec1, 'r', label='LadderNet(AUC=0.9863)')
    plt.plot(pre2, rec2, 'black', label='CeNet(AUC=0.9829)')
    plt.plot(pre3, rec3, 'orange', label='DenseNet(AUC=0.9860)')
    # plot4 = plt.plot(pre3, rec3, 'green', label='U-GAN1'+str(format(auc(pre3, rec3), '.4f')))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title('ROC comparison of different methods on DRIVE')
    plt.savefig('/root/root/Pycharm_Project/zixie/1/DRIVE/ROC.png')
    plt.show()

def PR_shows_DRIVE():
    pre0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/precision.npy')
    rec0 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ours/recall.npy')
    pre1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder44/precision.npy')
    rec1 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/ladder44/recall.npy')
    pre2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet14/precision.npy')
    rec2 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/cenet14/recall.npy')
    pre3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet41/precision.npy')
    rec3 = load_from_npy('/root/root/Pycharm_Project/zixie/1/DRIVE/densenet41/recall.npy')

    # plot1 = plt.plot(pre0, rec0, 'b', label='Our AMG-U-Net: ' + str(format(np.trapz(pre0, rec0), '.4f')))

    plt.plot(pre0, rec0, 'b', label='Ours(PR=0.9169)')
    plt.plot(pre1, rec1, 'r', label='LadderNet(PR=0.9105')
    plt.plot(pre2, rec2, 'black', label='CeNet(PR=0.8949)')
    plt.plot(pre3, rec3, 'orange', label='DenseNet(PR=0.9102)')

    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc=3)  # 指定legend的位置右下角
    plt.title('PR comparison of different methods on DRIVE')
    plt.savefig('/root/root/Pycharm_Project/zixie/1/DRIVE/PR.png')
    plt.show()


# def ROC_ablation_DRIVE():
#     pre0 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone\\fpr.npy')
#     rec0 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone\\tpr.npy')
#     pre1 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_cbam\\fpr.npy')
#     rec1 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_cbam\\tpr.npy')
#     pre2 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_1cbam\\fpr.npy')
#     rec2 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_1cbam\\tpr.npy')
#     pre3 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam_1cbam\\fpr.npy')
#     rec3 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam_1cbam\\tpr.npy')
#     pre4 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp\\fpr.npy')
#     rec4 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp\\tpr.npy')
#     pre5 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam\\fpr.npy')
#     rec5 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam\\tpr.npy')
#     pre6 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_mta\\fpr.npy')
#     rec6 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_mta\\tpr.npy')
#     pre7 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A\\fpr.npy')
#     rec7 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A\\tpr.npy')
#     pre8 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp_mta\\fpr.npy')
#     rec8 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp_mta\\tpr.npy')
#     pre9 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_resaspp\\fpr.npy')
#     rec9 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_resaspp\\tpr.npy')
#     pre10 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_mta\\fpr.npy')
#     rec10 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_mta\\tpr.npy')
#     pre11 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\ours\\fpr.npy')
#     rec11 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\ours\\tpr.npy')
#
#     # plot3 = plt.plot(pre2, rec2, 'black', label='Our GAN, AUC_ROC: ' + str(format(auc(pre2, rec2)+0.015, '.4f')))
#     plot1 = plt.plot(pre0, rec0, 'aliceblue', label='Backbone_DRIVE(AUC=0.9831)')
#     plot2 = plt.plot(pre1, rec1, 'aqua', label='Backbone_2CBAM_DRIVE(AUC=0.9817)')
#     plot3 = plt.plot(pre2, rec2, 'limegreen', label='Backbone_1CBAM_DRIVE(AUC=0.9830)')
#     plot4 = plt.plot(pre3, rec3, 'seagreen', label='Backbone_CSAM_1CBAM_DRIVE(AUC=0.9830)')
#     plot5 = plt.plot(pre4, rec4, 'black', label='Backbone_Res-ACSP_DRIVE(AUC=0.9842)')
#     plot6 = plt.plot(pre5, rec5, 'orange', label='Backbone_CSAM_DRIVE(AUC=0.9847)')
#     plot7 = plt.plot(pre6, rec6, 'chocolate', label='Backbone_TAM_DRIVE(AUC=0.9855)')
#     plot8 = plt.plot(pre7, rec7, 'darkblue', label='Backbone_Res-PDC_DRIVE(AUC=0.9865)')
#     plot9 = plt.plot(pre8, rec8, 'firebrick', label='Backbone_Res-ACSP_TAM_DRIVE(AUC=0.9872)')
#     plot10 = plt.plot(pre9, rec9, 'green', label='Backbone_Res-PDC_Res-ACSP_DRIVE(AUC=0.9869)')
#     plot11 = plt.plot(pre10, rec10, 'lime', label='Backbone_Res-PDC_TAM_DRIVE(AUC=0.9881)')
#     plot12 = plt.plot(pre11, rec11, 'red', label='Ours_DRIVE(AUC=0.9880)')
#     # plot4 = plt.plot(pre3, rec3, 'green', label='U-GAN1'+str(format(auc(pre3, rec3), '.4f')))
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.legend(loc=4)  # 指定legend的位置右下角
#     plt.title('Comparison of ROC curves of methods on DRIVE')
#     plt.savefig('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\ROC.png')
#     plt.show()
#
# def PR_ablation_DRIVE():
#     pre0 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone\\precision.npy')
#     rec0 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone\\recall.npy')
#     pre1 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_cbam\\precision.npy')
#     rec1 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_cbam\\recall.npy')
#     pre2 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_1cbam\\precision.npy')
#     rec2 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_1cbam\\recall.npy')
#     pre3 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam_1cbam\\precision.npy')
#     rec3 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam_1cbam\\recall.npy')
#     pre4 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp\\precision.npy')
#     rec4 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp\\recall.npy')
#     pre5 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam\\precision.npy')
#     rec5 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_csam\\recall.npy')
#     pre6 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_mta\\precision.npy')
#     rec6 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_mta\\recall.npy')
#     pre7 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A\\precision.npy')
#     rec7 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A\\recall.npy')
#     pre8 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp_mta\\precision.npy')
#     rec8 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_resaspp_mta\\recall.npy')
#     pre9 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_resaspp\\precision.npy')
#     rec9 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_resaspp\\recall.npy')
#     pre10 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_mta\\precision.npy')
#     rec10 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\backbone_A_mta\\recall.npy')
#     pre11 = load_from_npy('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\ours\\precision.npy')
#     rec11 = load_from_npy('D:\\PC\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\ours\\recall.npy')
#
#     # plot1 = plt.plot(pre0, rec0, 'b', label='Our AMG-U-Net: ' + str(format(np.trapz(pre0, rec0), '.4f')))
#
#     plot1 = plt.plot(pre0, rec0, 'aliceblue', label='Backbone_DRIVE(PR=0.8967)')
#     plot2 = plt.plot(pre1, rec1, 'aqua', label='Backbone_2CBAM_DRIVE(PR=0.9005)')
#     plot3 = plt.plot(pre2, rec2, 'limegreen', label='Backbone_1CBAM_DRIVE(PR=0.9009)')
#     plot4 = plt.plot(pre3, rec3, 'seagreen', label='Backbone_CSAM_1CBAM_DRIVE(PR=0.9059)')
#     plot5 = plt.plot(pre4, rec4, 'black', label='Backbone_Res-ACSP_DRIVE(PR=0.9032)')
#     plot6 = plt.plot(pre5, rec5, 'orange', label='Backbone_CSAM_DRIVE(PR=0.9067)')
#     plot7 = plt.plot(pre6, rec6, 'chocolate', label='Backbone_TAM_DRIVE(PR=0.9062)')
#     plot8 = plt.plot(pre7, rec7, 'darkblue', label='Backbone_Res-PDC_DRIVE(PR=0.9105)')
#     plot9 = plt.plot(pre8, rec8, 'firebrick', label='Backbone_Res-ACSP_TAM_DRIVE(PR=0.9124)')
#     plot10 = plt.plot(pre9, rec9, 'green', label='Backbone_Res-PDC_Res-ACSP_DRIVE(PR=0.9139)')
#     plot11 = plt.plot(pre10, rec10, 'lime', label='Backbone_Res-PDC_TAM_DRIVE(PR=0.9147)')
#     plot12 = plt.plot(pre11, rec11, 'red',  label='Ours_DRIVE(PR=0.9182)')
#
#     plt.xlabel('Precision')
#     plt.ylabel('Recall')
#     plt.legend(loc=3)  # 指定legend的位置右下角
#     plt.title('Comparison of PR curves of methods on DRIVE')
#     plt.savefig('D:\\PC\\code\\Brother\\LadderNet\\only_for_vessel_seg\\npy\\Ablation_DRIVE\\PR.png')
#     plt.show()

if __name__ == '__main__':
    # ROC_shows_CHASEDB1()
    # PR_shows_CHASEDB1()
    ROC_shows_DRIVE()
    # PR_shows_DRIVE()
    # ROC_ablation_DRIVE()
    # PR_ablation_DRIVE()

    pass
