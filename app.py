import pandas as pd
from h2o_wave import main, app, Q, ui, data
from sklearn.ensemble import RandomForestClassifier

# Global variables to store the data
df = None
test_df = None
train = None
test = None


@app('/waveApp')
async def serve(q: Q):
    global df, test_df, train, test

    # Initialization and setting up the UI
    if not q.client.initialized:
        setup_ui(q)
    # Handling user actions when uploading the dataset
    elif q.args.upload:
        upload_dataset(q)
    # Previewing the dataset
    elif q.args.preview:
        preview_dataset(q)
    # Handling user actions after uploading the dataset
    elif q.args.user_files:
        file_path = await q.site.download(q.args.user_files[0], '.')
        df = load_data(file_path)
        print(df.columns)
        print('Preview Dataset')
        print(df.head(10))
        preview_dataset(q)
    # Plotting data distribution for specific columns
    elif q.args.plot:
        plot_view(q)
    # Preprocessing the data
    elif q.args.preprocess:
        preprocessing_data(q)
    # Training the machine learning model
    elif q.args.train:
        train_model(q)
    else:
        upload_dataset(q)
    await q.page.save()


# Function to load data from CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Function to load the test dataset from CSV file
def test_data():
    tdata = pd.read_csv('test.csv')
    return tdata


# Function to set up the initial UI components
def setup_ui(q):
    q.page["header"] = ui.header_card(
        box="1 1 12 1",
        title='Wave App',
        subtitle='H2O.ai Technical Assignment',
    )

    q.page["tabs"] = ui.tab_card(
        box="1 2 9 1",
        items=[
            ui.tab(name="upload", label="Upload Dataset"),
            ui.tab(name="preview", label="Preview Dataset"),
            ui.tab(name="plot", label="Plot View"),
            ui.tab(name="preprocess", label="Preprocess Data"),
            ui.tab(name="train", label="Train and Predict Model"),
        ],
    )

    q.page['footer'] = ui.footer_card(
        box='1 10 12 1',
        caption='**Developed by Himidiri Himakanika**',
    )

    q.client.initialized = True


# Function to create a markdown table from DataFrame
def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row(['---'] * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])


# Function to create a markdown table for data preview
def make_preprocess_table(df):
    markdown_table = make_markdown_table(fields=df.columns.tolist(), rows=df.head(10).values.tolist())
    return markdown_table


# Function to handle dataset upload
def upload_dataset(q):
    del q.page['preview_dataset']
    del q.page['plot_view']
    del q.page['education_plot']
    del q.page['occupation_plot']
    del q.page['age_plot']
    del q.page['target_vs_age_plot']
    del q.page['target_vs_hours_plot']
    del q.page['preprocessing_data']
    del q.page['train_model']
    q.page['upload_dataset'] = ui.form_card(
        box='1 3 12 7',
        items=[
            ui.text_l('**UPLOAD DATASET**'),
            ui.file_upload(name='user_files', label='Upload census_data.csv', multiple=False),
        ],
    )


# Function to display a preview of the dataset
def preview_dataset(q):
    del q.page['upload_dataset']
    del q.page['plot_view']
    del q.page['education_plot']
    del q.page['occupation_plot']
    del q.page['age_plot']
    del q.page['target_vs_age_plot']
    del q.page['target_vs_hours_plot']
    del q.page['preprocessing_data']
    del q.page['train_model']
    if df is None:
        q.page['preview_dataset'] = ui.form_card(
            box='1 3 12 7',
            items=[
                ui.text_l('**PREVIEW DATASET**'),
                ui.text('Please upload the dataset first.'),
            ],
        )
        return
    q.page['preview_dataset'] = ui.form_card(
        box='1 3 12 7',
        items=[
            ui.text_l('**PREVIEW DATASET**'),
            ui.text(make_markdown_table(fields=df.columns.tolist(), rows=df.head(5000).values.tolist()))
        ],
    )


# Function to plot data distribution for specific columns
def plot_view(q):
    del q.page['upload_dataset']
    del q.page['preview_dataset']
    del q.page['plot_view']
    del q.page['preprocessing_data']
    del q.page['train_model']
    if df is None:
        q.page['plot_view'] = ui.form_card(
            box='1 3 12 7',
            items=[
                ui.text_l('**PLOT VIEW**'),
                ui.text('Please upload the dataset first.'),
            ],
        )
        return

    # Data distribution for education, occupation, and age
    education_distribution = df.groupby([' education', ' sex']).size().reset_index(name='count')
    occupation_distribution = df.groupby([' occupation', ' sex']).size().reset_index(name='count')
    age_distribution = df.groupby(['age', ' sex']).size().reset_index(name='count')
    target_vs_age = df.groupby(['age', ' amount']).size().reset_index(name='count')
    target_vs_hours = df.groupby([' hour-per-week', ' amount']).size().reset_index(name='count')

    q.page['education_plot'] = ui.plot_card(
        box='1 3 4 3',
        title='Education Distribution',
        data=data(fields=['education', 'sex', 'count'], rows=education_distribution.values.tolist()),
        plot=ui.plot([
            ui.mark(type='interval',
                    x='=education', y='=count',
                    color='=sex')
        ])
    )

    q.page['occupation_plot'] = ui.plot_card(
        box='5 3 4 3',
        title='Occupation Distribution',
        data=data(fields=['occupation', 'sex', 'count'], rows=occupation_distribution.values.tolist()),
        plot=ui.plot([
            ui.mark(type='interval',
                    x='=occupation', y='=count',
                    color='=sex')
        ])
    )

    q.page['age_plot'] = ui.plot_card(
        box='9 3 4 3',
        title='Age Distribution',
        data=data(fields=['age', 'sex', 'count'], rows=age_distribution.values.tolist()),
        plot=ui.plot([ui.mark(type='line', x='=age', y='=count', color='=sex')]),
    )

    q.page['target_vs_age_plot'] = ui.plot_card(
        box='3 6 4 3',
        title='Amount vs Age',
        data=data(fields=['age', 'amount', 'count'], rows=target_vs_age.values.tolist()),
        plot=ui.plot(
            [ui.mark(type='point', x='=age', y='=count', size='=count', color='=amount', shape='circle')]),
    )

    q.page['target_vs_hours_plot'] = ui.plot_card(
        box='7 6 4 3',
        title='Amount vs Hours Per Week',
        data=data(fields=['hour-per-week', 'amount', 'count'], rows=target_vs_hours.values.tolist()),
        plot=ui.plot([ui.mark(type='point', x='=hour-per-week', y='=amount', size='=count', color='=amount',
                              shape='circle')]),
    )


# Function to preprocess the data
def preprocess(df):
    df = df.replace('?', pd.NA).dropna()
    df = pd.get_dummies(df, columns=[' workclass', ' education', ' marital-status', ' occupation', ' relationship',
                                     ' race', ' sex', ' native-country'])
    df[' amount'] = df[' amount'].map({' <=50K': 0, ' >50K': 1})
    df.drop(' fnlwgt', axis=1, inplace=True)
    return df


# Function to handle data preprocessing
def preprocessing_data(q):
    del q.page['upload_dataset']
    del q.page['preview_dataset']
    del q.page['plot_view']
    del q.page['education_plot']
    del q.page['occupation_plot']
    del q.page['age_plot']
    del q.page['target_vs_age_plot']
    del q.page['target_vs_hours_plot']
    del q.page['preprocessing_data']
    del q.page['train_model']
    global df, test_df, train, test

    if df is None:
        q.page['preprocessing_data'] = ui.form_card(
            box='1 3 12 7',
            items=[
                ui.text_l('**PREPROCESS DATA**'),
                ui.text('Please upload the dataset first.'),
            ],
        )
        return
    test_df = test_data()
    train = preprocess(df)
    test = preprocess(test_df)
    train_data_table = make_preprocess_table(train)
    print('Preprocess Uploaded Dataset')
    print(train.head(10))
    print('Preprocess Test Dataset')
    print(test.head(10))

    q.page['preprocessing_data'] = ui.form_card(
        box='1 3 12 7',
        items=[
            ui.text_l('**PREPROCESSED DATA**'),
            ui.text(train_data_table),
        ],
    )


# Function to train model
def random_forest_training(train, test):
    y = train[' amount']
    train = train.drop([' amount'], axis=1)
    X = train.values
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X, y)
    return clf


# Function to get information about the trained model
def get_model_info(clf):
    model_info = {
        'Feature Importances': clf.feature_importances_,
        'Number of Estimators': clf.n_estimators,
        'Max Depth': clf.max_depth,
        'Number of Classes': clf.n_classes_,
    }
    return model_info


# Function to train the machine learning model
def train_model(q):
    del q.page['upload_dataset']
    del q.page['preview_dataset']
    del q.page['plot_view']
    del q.page['education_plot']
    del q.page['occupation_plot']
    del q.page['age_plot']
    del q.page['target_vs_age_plot']
    del q.page['target_vs_hours_plot']
    del q.page['preprocessing_data']
    del q.page['train_model']
    global df, train, test

    if df is not None:
        q.page['train_model'] = ui.form_card(box='1 12 5 2', items=[ui.progress('Running...')])
        model = random_forest_training(train, test)
        model_info = get_model_info(model)

        # Message to indicate successful model training
        message = 'Model Trained Successfully'
        field_names = list(model_info.keys())
        rows = [list(model_info.values())]

        markdown_table = make_markdown_table(fields=field_names, rows=rows)

        q.page['train_model'] = ui.form_card(
            box='1 3 12 7',
            items=[
                ui.text_l('**TRAIN AND PREDICT MODEL**'),
                ui.message_bar('info', message),
                ui.text(markdown_table),
            ],
        )
    else:
        q.page['train_model'] = ui.form_card(
            box='1 3 12 7',
            items=[
                ui.text_l('**TRAIN AND PREDICT MODEL**'),
                ui.text('Please preprocess data before training the model'),
            ],
        )
