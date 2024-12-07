#ifndef COURSES_H
#define COURSES_H

#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QListWidget>
#include <QMap>
#include <QString>
#include <QInputDialog>
#include <QMessageBox>
#include <QComboBox>

QT_BEGIN_NAMESPACE
namespace Ui {
class Courses;
}
QT_END_NAMESPACE

class Courses : public QWidget
{
    Q_OBJECT

public:
    explicit Courses(QWidget *parent = nullptr);
    ~Courses();

    QMap<QString, QStringList> getCourses() const { return courses; } // 提供访问 courses 的方法
    QMap<QString, QStringList>& getCoursesMutable();
    void create();

signals:
    void fileUpdated(const QString &selectedCourse, const QString &fileName);


private slots:

    void deleteCourse(QPushButton *courseButton);

    void on_logoutButton_clicked();

    void on_switchButton_clicked();

    QString on_addButton_clicked();

    void on_deleteButton_clicked();

    QString on_searchButton_click();

    void on_finishDeleteButton_clicked();

    void on_searchButton_check();

    void on_okButton_clicked();

    void onFileButtonClicked(QPushButton *fileButton);

private:
    Ui::Courses *ui;

    QVBoxLayout *courseLayout;     // 左侧课程按钮布局
    QVBoxLayout *workspaceLayout;  // 工作区域布局
    QListWidget *resultList;       // 搜索结果列表
    QVector<QPushButton*> courseButtons;
    QWidget * selectCoursePage ;
    QComboBox *tagComboBox;

    QMap<QString, QStringList> courses; // 存储课程和文件的映射
    bool deleteMode;               // 删除模式标志
    Courses *coursesWindow;
};


#endif // COURSES_H
