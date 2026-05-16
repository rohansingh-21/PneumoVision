import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, history_fine):
    split = len(history.history['accuracy'])

    acc      = history.history['accuracy']  + history_fine.history['accuracy']
    val_acc  = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss     = history.history['loss']      + history_fine.history['loss']
    val_loss = history.history['val_loss']  + history_fine.history['val_loss']
    auc      = history.history['auc']       + history_fine.history['auc']
    val_auc  = history.history['val_auc']   + history_fine.history['val_auc']

    for values, val_values, title, ylabel in [
        (acc,  val_acc,  'Model Accuracy', 'Accuracy'),
        (loss, val_loss, 'Model Loss',     'Loss'),
        (auc,  val_auc,  'Model AUC',      'AUC'),
    ]:
        plt.plot(values)
        plt.plot(val_values)
        plt.axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Fine-tune start'])
        plt.show()


def evaluate(model, test_generator):
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
