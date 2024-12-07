#include "courses.h"
#include "ui_courses.h"
#include "ui_coursespace.h"
#include "ui_courses.h"
#include "mainwindow.h"
#include "coursespace.h"
#include <QDebug> // 测试使用

Courses::Courses(QWidget *parent)
    :QWidget(parent)
    ,ui(new Ui::Courses)
    ,courseLayout(nullptr)
    ,selectCoursePage(nullptr)
    ,deleteMode(false)
    ,coursesWindow(nullptr)
{
    ui->setupUi(this);
    qDebug() << coursesWindow;
    courseLayout = qobject_cast<QVBoxLayout *>(ui->courseArea->layout());
        if (!courseLayout) {
            courseLayout = new QVBoxLayout();
            ui->courseArea->setLayout(courseLayout);
        }

    // 按钮事件绑定
    connect(ui->searchButton, &QLineEdit::textChanged, this, &Courses::on_searchButton_check);
    //connect(ui->logoutButton, &QPushButton::clicked, this, &Courses::on_logoutButton_clicked);
    //connect(ui->switchButton, &QPushButton::clicked, this, &Courses::on_switchButton_clicked);
    connect(ui->searchButton, &QLineEdit::returnPressed, this, &Courses::on_searchButton_click);
    connect(ui->okButton, &QPushButton::clicked, this, &Courses::on_searchButton_click);


}

Courses::~Courses()
{
    delete ui;
}

void Courses:: create()
{
    this->show();
}

void Courses::on_logoutButton_clicked()
{
    MainWindow *mainWindow = new MainWindow();
    mainWindow->show();
    this->close();
}

void Courses::on_switchButton_clicked()
{
    on_logoutButton_clicked();
}

QString Courses::on_addButton_clicked()
{   bool ok;
    QString courseName = QInputDialog::getText(this, "add course", "Please enter the course name：", QLineEdit::Normal, "", &ok);
    if (ok && !courseName.isEmpty()) {
        if (courses.contains(courseName)) {
            QMessageBox::warning(this, "error", "This course already exits！");
            return "empty";
        }

        // 创建课程按钮
        QPushButton *courseButton = new QPushButton(courseName);
        courseButton->setStyleSheet("background-color: rgb(255, 255, 255);");
        courseLayout->addWidget(courseButton);
        connect(courseButton, &QPushButton::clicked, [this, courseName]() {
        CourseSpace *courseSpace = new CourseSpace(nullptr,this, courseName);
        //qDebug() << coursesWindow;
        courseSpace->setWindowTitle(courseName); // 设置窗口标题为课程名称
        courseSpace->setCourseName(courseName); // 动态设置课程名称
        //create_workspace(courseName);//AI接口
        courseSpace->onFileUpdated(courseName);
        courseSpace->show();
         //监听返回信号
        connect(courseSpace, &CourseSpace::backButtonClicked, this, [this, courseSpace]() {
            this->show(); // 显示 Courses 界面
            courseSpace->close();
        });
        this->hide();// 显示课程空间页面
        });

        courses[courseName] = QStringList(); // 初始化课程文件列表
        }
    return courseName;
}

void Courses::deleteCourse(QPushButton *courseButton)
{
    // 删除对应的课程按钮
    courseLayout->removeWidget(courseButton);  // 从布局中移除
    courseButton->deleteLater();  // 删除该按钮

    // 删除课程数据
    QString courseName = courseButton->text();
    courses.remove(courseName);  // 从课程列表中移除该课程
}

void Courses::on_deleteButton_clicked()
{
    // 切换删除模式
    deleteMode = !deleteMode;

    // 记录原始课程按钮
    QVector<QPushButton*> courseButtons;
    for (int i = 0; i < courseLayout->count(); ++i) {
        QWidget *widget = courseLayout->itemAt(i)->widget();
        if (widget && widget->inherits("QPushButton")) {
            QPushButton *courseButton = qobject_cast<QPushButton *>(widget);
            courseButtons.append(courseButton);
        }
    }

    // 清空布局（暂存课程按钮）
    for (QPushButton *courseButton : courseButtons) {
        courseLayout->removeWidget(courseButton);
    }

    // 重新布局课程按钮并添加删除按钮
    for (QPushButton *courseButton : courseButtons) {
        if (deleteMode) {
            // 创建一个新的水平布局
            QWidget *buttonContainer = new QWidget();
            QHBoxLayout *hLayout = new QHBoxLayout(buttonContainer);
            hLayout->setContentsMargins(0, 0, 0, 0);

            // 将课程按钮添加到左侧
            hLayout->addWidget(courseButton);

            // 创建删除按钮并添加到右侧
            QPushButton *deleteButton = new QPushButton("X", this);
            deleteButton->setFixedSize(30, courseButton->height()); // 调整删除按钮的大小
            deleteButton->setStyleSheet("background-color: rgb(0, 0, 0); color: white;");
            hLayout->addWidget(deleteButton);

            // 连接删除按钮的点击事件
            connect(deleteButton, &QPushButton::clicked, this, [this, courseButton, buttonContainer]() {
                deleteCourse(courseButton);
                buttonContainer->deleteLater(); // 删除包装容器
            });

            // 添加容器到主布局
            courseLayout->addWidget(buttonContainer);
        } else {
            // 在非删除模式下仅添加课程按钮
            courseLayout->addWidget(courseButton);
        }
    }
}


void Courses::on_finishDeleteButton_clicked()
{
    deleteMode = false; // 关闭删除模式
    qDebug() << "Finish Delete clicked. Exiting delete mode.";
    qDebug() << "Current layout count:" << courseLayout->count();

    QVector<QPushButton*> courseButtons;

    // 遍历布局中的所有容器
    for (int i = 0; i < courseLayout->count(); ++i) {
        QWidget *buttonContainer = courseLayout->itemAt(i)->widget();
        if (!buttonContainer) continue; // 忽略空项

        QHBoxLayout *hLayout = qobject_cast<QHBoxLayout *>(buttonContainer->layout());
        if (!hLayout) continue;

        // 查找课程按钮并重新保存
        for (int j = 0; j < hLayout->count(); ++j) {
            QWidget *widget = hLayout->itemAt(j)->widget();
            QPushButton *button = qobject_cast<QPushButton *>(widget);
            if (!button) continue;

            // 判断是否为课程按钮
            if (button->text() != "X") {
                courseButtons.append(button);
            }
        }

        // 删除整个容器
        courseLayout->removeWidget(buttonContainer);
        buttonContainer->deleteLater();
    }

    // 重建课程按钮布局（清除删除按钮）
    for (QPushButton *courseButton : courseButtons) {
        courseLayout->addWidget(courseButton);
    }
}

QMap<QString, QStringList>& Courses::getCoursesMutable() {
    return courses;
}



void Courses::on_searchButton_check()
{
    // 获取输入框中的文本
    QString search_context = ui->searchButton->text();
    QString originalContent = "Search files:";

    // 确保输入框的内容不会被意外修改
    if (!search_context.startsWith(originalContent)) {
        ui->searchButton->setText(originalContent);
        QMessageBox::warning(this, "Error", "You cannot modify the original content!");
    }
}


// 槽函数
QString Courses::on_searchButton_click()
{
    // 获取输入框中的文本
    QString search_context = ui->searchButton->text();
    QString originalContent = "Search files:";

    // 提取用户输入的内容（去掉前缀）
    QString inputContent = search_context.mid(originalContent.length()).trimmed();

    // 将输入以空格分隔存储到 std::vector<std::string> 中
    std::vector<std::string> newContent;
    for (const QString &word : inputContent.split(" ", QString::SkipEmptyParts)) {
        newContent.push_back(word.toStdString());
    }

    // 打印分隔后的内容（用于调试）
    qDebug() << "Parsed search content:" << inputContent;
    for (const std::string &str : newContent) {
        qDebug() << QString::fromStdString(str);
    }

    // 清空已有的文件按钮
    QLayout *layout = ui->searchfilelayout;
    QLayoutItem *item;
    while ((item = layout->takeAt(0)) != nullptr) {
        // 删除所有布局中的子控件（按钮）
        QWidget *widget = item->widget();
        if (widget) {
            delete widget; // 删除文件按钮
        }
        delete item; // 删除布局项
    }
     ui->searchButton->setText(originalContent);  // 重新设置为原始内容，清空用户输入


    //调用搜索函数，获得搜索结果，解析搜索结果，用循环把每个文件都放到界面上
    ui->text1->hide();
    ui->text2->hide();
    // 调用搜索函数，获取文件名列表
    //std::vector<std::string> fileNames = search_file(newContent);
    std::vector<std::string>fileNames = {"aha", "haha", "xiha"};

    // 遍历文件名，创建按钮
    for (const std::string &fileName : fileNames) {
        QString fileNameQString = QString::fromStdString(fileName);

        // 创建文件按钮
        QPushButton *fileButton = new QPushButton(fileNameQString, this);
        fileButton->setStyleSheet("background-color: rgb(255, 255, 255);");
        fileButton->setProperty("fileName", fileNameQString);

        // 添加按钮到布局中
        ui->searchfilelayout->addWidget(fileButton);

        // 连接按钮点击事件
        connect(fileButton, &QPushButton::clicked, [this, fileButton]() {
            onFileButtonClicked(fileButton);
        });
    }


    return  inputContent;

}

void Courses::onFileButtonClicked(QPushButton *fileButton)
{
    QString fileName = fileButton->property("fileName").toString();
    // 创建上传页面
    selectCoursePage = new QWidget(this);
    selectCoursePage->setWindowTitle("Upload File");
    selectCoursePage->setStyleSheet("background-color: rgb(68, 98, 162); color: white;");
    selectCoursePage->resize(400, 300);

    QVBoxLayout *layout = new QVBoxLayout(selectCoursePage);

    // 标签选择
    QLabel *tagLabel = new QLabel("Choose a course space:", selectCoursePage);
    tagComboBox = new QComboBox(selectCoursePage);
    if (courses.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please create a course space first!");
        return;
    }
    // 遍历 courses，将课程名添加到 QComboBox
    for (const QString &courseName : courses.keys()) {
        tagComboBox->addItem(courseName);  // 添加到 tagComboBox
    }

    layout->addWidget(tagLabel);
    layout->addWidget(tagComboBox);
    // 添加确认按钮
    QPushButton *confirmButton = new QPushButton("Confirm", selectCoursePage);
    confirmButton->setStyleSheet("background-color: white; color: black; padding: 5px;");
    layout->addWidget(confirmButton);


    // 连接确认按钮的点击事件
    connect(confirmButton, &QPushButton::clicked, [this, fileName]() {
    QString selectedCourse = tagComboBox->currentText();  // 获取选中的课程名称
    qDebug() << "Selected course:" << selectedCourse;

    // 将文件名添加到选中的课程
    if (!selectedCourse.isEmpty()) {
        courses[selectedCourse].append(fileName);//画一下按钮；话说filename有必要添加到这个list里面吗
//        QPushButton *downloadfileButton = new QPushButton(fileName, this);

//        downloadfileButton->setStyleSheet("background-color: rgb(255, 255, 255); color: black;");

        emit fileUpdated(selectedCourse, fileName);
    qDebug()<<"coures"<<courses;
    }

        // 关闭弹窗
        selectCoursePage->close();
    });
selectCoursePage->show();
}

void Courses::on_okButton_clicked()
{
    on_searchButton_click();
}
