import pandas as pd
import numpy as np
import json
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, LSTM, Reshape,Dropout
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping
import random



# Load JSON data
file_path = 'F:\GenzDealz\output_dataset.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.json_normalize(data)

# Drop empty rows
df.dropna(inplace=True)

# Convert 'Purchase Date' to datetime with error handling
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d-%m-%Y %H:%M', errors='coerce')

# Drop rows with invalid dates
df.dropna(subset=['Purchase Date'], inplace=True)

# Sort data by Customer ID and Purchase Date
df.sort_values(by=['Customer ID', 'Purchase Date'], inplace=True)

# Encode categorical features
le_customer = LabelEncoder()
le_product_category = LabelEncoder()
le_company = LabelEncoder()

df['Customer ID'] = le_customer.fit_transform(df['Customer ID'])
df['Product Category'] = le_product_category.fit_transform(df['Product Category'])
df['Company'] = le_company.fit_transform(df['Company'])

# Calculate quantity and frequency
df['Purchase Count'] = df.groupby(['Customer ID', 'Product Category'])['Quantity'].transform('sum')
df['Purchase Frequency'] = df.groupby(['Customer ID', 'Product Category'])['Purchase Date'].transform('count')


agg_df = df.groupby(['Customer ID', 'Product Category', 'Company']).agg({
    'Purchase Count': 'max',
    'Purchase Frequency': 'max'
}).reset_index()


num_customers = agg_df['Customer ID'].nunique()
num_products = agg_df['Product Category'].nunique()
num_companies = agg_df['Company'].nunique()

customer_input = Input(shape=(1,), name='customer_input')
product_input = Input(shape=(1,), name='product_input')
company_input = Input(shape=(1,), name='company_input')

customer_embedding = Embedding(input_dim=num_customers, output_dim=50, name='customer_embedding')(customer_input)
product_embedding = Embedding(input_dim=num_products, output_dim=50, name='product_embedding')(product_input)
company_embedding = Embedding(input_dim=num_companies, output_dim=50, name='company_embedding')(company_input)

customer_vec = Flatten()(customer_embedding)
product_vec = Flatten()(product_embedding)
company_vec = Flatten()(company_embedding)

concat = Concatenate()([customer_vec, product_vec, company_vec])

# Additional input for Quantity
quantity_input = Input(shape=(1,), name='quantity_input')
quantity_reshape = Reshape((1, 1))(quantity_input)
quantity_lstm = LSTM(20, return_sequences=False, kernel_regularizer=l2(0.2))(quantity_reshape)

concat = Concatenate()([concat, quantity_lstm])

dense_1 = Dense(64, activation='relu', kernel_regularizer=l2(0.2))(concat)
dropout_1 = Dropout(0.5)(dense_1)  # Dropout after the first dense layer

dense_2 = Dense(32, activation='relu', kernel_regularizer=l2(0.2))(dropout_1)
dropout_2 = Dropout(0.5)(dense_2)  # Dropout after the second dense layer

dense_3 = Dense(16, activation='relu', kernel_regularizer=l2(0.2))(dropout_2)
dropout_3 = Dropout(0.5)(dense_3)  # Dropout after the third dense layer

output = Dense(1, activation='sigmoid')(dropout_3)

model = Model(inputs=[customer_input, product_input, company_input, quantity_input], outputs=output)


optimizer = 'adam'


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
####################################################################################################
# Prepare input data
customer_ids = agg_df['Customer ID'].values
product_ids = agg_df['Product Category'].values
company_ids = agg_df['Company'].values
quantities = agg_df['Purchase Count'].values


y = (agg_df['Purchase Count'] > 0).astype(int).values

# Split the data into training and testing sets (80% train, 20% test)
X_train_customer, X_test_customer, X_train_product, X_test_product, X_train_company, X_test_company, X_train_quantity, X_test_quantity, y_train, y_test = train_test_split(
    customer_ids, product_ids, company_ids, quantities, y, test_size=0.2, random_state=42)


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    [X_train_customer, X_train_product, X_train_company, X_train_quantity], y_train,
    epochs=10, batch_size=64,
    validation_data=([X_test_customer, X_test_product, X_test_company, X_test_quantity], y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate([X_test_customer, X_test_product, X_test_company, X_test_quantity], y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Select three random customers from the training set
random_customers = random.sample(list(agg_df['Customer ID'].unique()), 3)

################################################# RECOMMENDATIONS ####################################
# Generate recommendations
for customer in random_customers:
    customer_encoded = np.array([customer])
    recommendations = []

    for product in range(num_products):
        for company in range(num_companies):
            pred = model.predict([customer_encoded, np.array([product]), np.array([company]), np.array([1])])  # Assume quantity 1 for prediction
            recommendations.append((le_product_category.inverse_transform([product])[0], le_company.inverse_transform([company])[0], pred[0][0]))

    # Sort recommendations by predicted value
    recommendations.sort(key=lambda x: x[2], reverse=True)
    print(f"Recommendations for Customer {customer}:")
    for rec in recommendations[:5]:  # Print top 5 recommendations
        print(f"Product Category: {rec[0]}, Company: {rec[1]}, Score: {rec[2]:.4f}")