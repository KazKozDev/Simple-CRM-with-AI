import sqlite3
import datetime
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import logging
import time

# Configure logging
logging.basicConfig(
    filename='crm_app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class Database:
    def __init__(self, db_name: str = "crm.db"):
        try:
            self.conn = sqlite3.connect(db_name, check_same_thread=False)
            self.create_tables()
            logging.info("Database connection established successfully.")
        except Exception as e:
            logging.error(f"Database connection error: {str(e)}")
            st.error(f"Database connection error: {str(e)}")

    def create_tables(self):
        with self.conn:
            try:
                # Company Information
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS company_info (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        city TEXT,
                        industry TEXT,
                        target_market TEXT,
                        unique_selling_points TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Products/Services
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS products (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT CHECK(type IN ('Product', 'Service')),
                        description TEXT,
                        price REAL,
                        features TEXT,
                        target_audience TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Clients
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS clients (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT,
                        phone TEXT,
                        company TEXT,
                        industry TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Deals
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS deals (
                        id INTEGER PRIMARY KEY,
                        client_id INTEGER,
                        product_id INTEGER,
                        amount REAL,
                        status TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (client_id) REFERENCES clients (id),
                        FOREIGN KEY (product_id) REFERENCES products (id)
                    )
                ''')

                # Tasks
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        due_date DATE,
                        status TEXT,
                        client_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (client_id) REFERENCES clients (id)
                    )
                ''')

                # Client Analyses
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS client_analyses (
                        id INTEGER PRIMARY KEY,
                        client_id INTEGER,
                        analysis TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (client_id) REFERENCES clients (id)
                    )
                ''')

                # Deal Analyses
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS deal_analyses (
                        id INTEGER PRIMARY KEY,
                        deal_id INTEGER,
                        analysis TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (deal_id) REFERENCES deals (id)
                    )
                ''')
                logging.info("Database tables created successfully or already exist.")
            except Exception as e:
                logging.error(f"Error creating tables: {str(e)}")
                st.error(f"Error creating tables: {str(e)}")


class CRMAnalytics:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

    def _get_llm_response(self, prompt: str) -> str:
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=10  # seconds
                )
                response.raise_for_status()
                llm_response = response.json().get('response', "No response from AI model.")
                logging.info("AI response successfully received.")
                return llm_response
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        st.error("Unable to connect to AI analytics service after multiple attempts.")
        return "Analysis unavailable at the moment."

    def analyze_client_potential(self, client_data: Dict, company_info: Dict, products: List[Dict]) -> str:
        prompt = f"""
        Based on the following information, analyze the client's potential:

        Company Context:
        - Company: {company_info.get('name')}
        - Industry: {company_info.get('industry')}
        - City: {company_info.get('city')}
        - Target Market: {company_info.get('target_market')}
        - Unique Selling Points: {company_info.get('unique_selling_points')}

        Client Information:
        - Name: {client_data.get('name')}
        - Industry: {client_data.get('industry')}
        - Company: {client_data.get('company')}
        - Notes: {client_data.get('notes')}

        Available Products/Services:
        {json.dumps(products, indent=2)}

        Please provide a comprehensive analysis:
        1. Client-Product Fit
        2. Probability of successful deals
        3. Recommended products/services for this client
        4. Suggested approach and next actions
        5. Potential deal value estimation
        """
        return self._get_llm_response(prompt)

    def predict_deal_success(self, deal_data: Dict, client_data: Dict, product_data: Dict, company_info: Dict) -> str:
        prompt = f"""
        Analyze the probability of successfully closing this deal based on the following context:

        Company Information:
        - Company: {company_info.get('name')}
        - Industry: {company_info.get('industry')}
        - City: {company_info.get('city')}
        - Target Market: {company_info.get('target_market')}
        - Unique Selling Points: {company_info.get('unique_selling_points')}

        Client Information:
        - Name: {client_data.get('name')}
        - Industry: {client_data.get('industry')}
        - Company: {client_data.get('company')}

        Product/Service Information:
        - Name: {product_data.get('name')}
        - Type: {product_data.get('type')}
        - Price: {product_data.get('price')}
        - Target Audience: {product_data.get('target_audience')}

        Deal Information:
        - Amount: {deal_data.get('amount')}
        - Description: {deal_data.get('description')}

        Please provide:
        1. Success probability percentage
        2. Key risk factors
        3. Competitive advantages
        4. Recommended negotiation strategy
        5. Value proposition alignment
        6. Specific actions to increase success chances
        """
        return self._get_llm_response(prompt)


class CRM:
    def __init__(self):
        self.db = Database()
        self.analytics = CRMAnalytics()
        self.DEAL_STATUSES = ["New", "In Progress", "Won", "Lost", "On Hold"]

    def save_company_info(self, data: Dict) -> bool:
        try:
            with self.db.conn:
                # Check if company info exists
                existing = self.db.conn.execute('SELECT id FROM company_info LIMIT 1').fetchone()

                if existing:
                    # Update
                    self.db.conn.execute('''
                        UPDATE company_info 
                        SET name=?, description=?, city=?, industry=?, target_market=?, 
                            unique_selling_points=?, updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    ''', (
                        data['name'], data['description'], data['city'], data['industry'],
                        data['target_market'], data['unique_selling_points'], existing[0]
                    ))
                    logging.info("Company information updated.")
                else:
                    # Insert
                    self.db.conn.execute('''
                        INSERT INTO company_info 
                        (name, description, city, industry, target_market, unique_selling_points)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        data['name'], data['description'], data['city'], data['industry'],
                        data['target_market'], data['unique_selling_points']
                    ))
                    logging.info("Company information added.")
            return True
        except Exception as e:
            logging.error(f"Error saving company information: {str(e)}")
            st.error(f"Error saving company information: {str(e)}")
            return False

    def get_company_info(self) -> Dict:
        try:
            result = self.db.conn.execute('''
                SELECT name, description, city, industry, target_market, unique_selling_points
                FROM company_info
                LIMIT 1
            ''').fetchone()

            if result:
                company_info = dict(zip(
                    ['name', 'description', 'city', 'industry', 'target_market', 'unique_selling_points'],
                    result
                ))
                logging.info("Company information retrieved.")
                return company_info
            logging.info("No company information found.")
            return {}
        except Exception as e:
            logging.error(f"Error retrieving company information: {str(e)}")
            st.error(f"Error retrieving company information: {str(e)}")
            return {}

    def add_product(self, data: Dict) -> int:
        try:
            with self.db.conn:
                cursor = self.db.conn.execute('''
                    INSERT INTO products 
                    (name, type, description, price, features, target_audience)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['name'], data['type'], data['description'],
                    data['price'], data['features'], data['target_audience']
                ))
                product_id = cursor.lastrowid
                logging.info(f"Product/Service added with ID: {product_id}.")
                return product_id
        except Exception as e:
            logging.error(f"Error adding product/service: {str(e)}")
            st.error(f"Error adding product/service: {str(e)}")
            return -1

    def get_products(self) -> List[Dict]:
        try:
            cursor = self.db.conn.execute('SELECT * FROM products')
            columns = [description[0] for description in cursor.description]
            products = [dict(zip(columns, row)) for row in cursor.fetchall()]
            logging.info(f"{len(products)} products/services retrieved.")
            return products
        except Exception as e:
            logging.error(f"Error retrieving products/services: {str(e)}")
            st.error(f"Error retrieving products/services: {str(e)}")
            return []

    def get_product_by_id(self, product_id: int) -> Dict:
        try:
            cursor = self.db.conn.execute('SELECT * FROM products WHERE id = ?', (product_id,))
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            product = dict(zip(columns, row)) if row else {}
            if product:
                logging.info(f"Product retrieved by ID: {product_id}.")
            else:
                logging.warning(f"Product with ID {product_id} not found.")
            return product
        except Exception as e:
            logging.error(f"Error retrieving product by ID {product_id}: {str(e)}")
            st.error(f"Error retrieving product by ID {product_id}: {str(e)}")
            return {}

    def add_client(self, name: str, email: str = None, phone: str = None,
                  company: str = None, industry: str = None, notes: str = None) -> int:
        try:
            with self.db.conn:
                cursor = self.db.conn.execute('''
                    INSERT INTO clients 
                    (name, email, phone, company, industry, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (name, email, phone, company, industry, notes))
                client_id = cursor.lastrowid
                logging.info(f"Client added with ID: {client_id}.")
                return client_id
        except Exception as e:
            logging.error(f"Error adding client: {str(e)}")
            st.error(f"Error adding client: {str(e)}")
            return -1

    def get_client_by_id(self, client_id: int) -> Dict:
        try:
            cursor = self.db.conn.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            client = dict(zip(columns, row)) if row else {}
            if client:
                logging.info(f"Client retrieved by ID: {client_id}.")
            else:
                logging.warning(f"Client with ID {client_id} not found.")
            return client
        except Exception as e:
            logging.error(f"Error retrieving client by ID {client_id}: {str(e)}")
            st.error(f"Error retrieving client by ID {client_id}: {str(e)}")
            return {}

    def save_client_analysis(self, client_id: int, analysis: str) -> bool:
        try:
            with self.db.conn:
                self.db.conn.execute('''
                    INSERT INTO client_analyses (client_id, analysis)
                    VALUES (?, ?)
                ''', (client_id, analysis))
                logging.info(f"Client analysis saved for ID {client_id}.")
            return True
        except Exception as e:
            logging.error(f"Error saving client analysis: {str(e)}")
            st.error(f"Error saving client analysis: {str(e)}")
            return False

    def save_deal_analysis(self, deal_id: int, analysis: str) -> bool:
        try:
            with self.db.conn:
                self.db.conn.execute('''
                    INSERT INTO deal_analyses (deal_id, analysis)
                    VALUES (?, ?)
                ''', (deal_id, analysis))
                logging.info(f"Deal analysis saved for ID {deal_id}.")
            return True
        except Exception as e:
            logging.error(f"Error saving deal analysis: {str(e)}")
            st.error(f"Error saving deal analysis: {str(e)}")
            return False

    def get_client_analysis(self, client_id: int) -> Optional[Dict]:
        try:
            cursor = self.db.conn.execute('''
                SELECT analysis, generated_at 
                FROM client_analyses 
                WHERE client_id = ? 
                ORDER BY generated_at DESC 
                LIMIT 1
            ''', (client_id,))
            result = cursor.fetchone()
            if result:
                analysis = {
                    'analysis': result[0],
                    'generated_at': result[1]
                }
                logging.info(f"Client analysis retrieved for ID {client_id}.")
                return analysis
            logging.info(f"No client analysis found for ID {client_id}.")
            return None
        except Exception as e:
            logging.error(f"Error retrieving client analysis: {str(e)}")
            st.error(f"Error retrieving client analysis: {str(e)}")
            return None

    def get_deal_analysis(self, deal_id: int) -> Optional[Dict]:
        try:
            cursor = self.db.conn.execute('''
                SELECT analysis, generated_at 
                FROM deal_analyses 
                WHERE deal_id = ? 
                ORDER BY generated_at DESC 
                LIMIT 1
            ''', (deal_id,))
            result = cursor.fetchone()
            if result:
                analysis = {
                    'analysis': result[0],
                    'generated_at': result[1]
                }
                logging.info(f"Deal analysis retrieved for ID {deal_id}.")
                return analysis
            logging.info(f"No deal analysis found for ID {deal_id}.")
            return None
        except Exception as e:
            logging.error(f"Error retrieving deal analysis: {str(e)}")
            st.error(f"Error retrieving deal analysis: {str(e)}")
            return None

    def regenerate_client_analysis(self, client_id: int) -> str:
        try:
            client_data = self.get_client_by_id(client_id)
            company_info = self.get_company_info()
            products = self.get_products()

            analysis = self.analytics.analyze_client_potential(
                client_data, company_info, products
            )
            self.save_client_analysis(client_id, analysis)
            logging.info(f"Client analysis regenerated for ID {client_id}.")
            return analysis
        except Exception as e:
            logging.error(f"Error regenerating client analysis: {str(e)}")
            st.error(f"Error regenerating client analysis: {str(e)}")
            return "Failed to regenerate analysis."

    def regenerate_deal_analysis(self, deal_id: int) -> Optional[str]:
        try:
            deal = self.db.conn.execute('SELECT * FROM deals WHERE id = ?', (deal_id,)).fetchone()
            if not deal:
                logging.warning(f"Deal with ID {deal_id} not found for analysis.")
                return None

            deal_data = dict(zip(['id', 'client_id', 'product_id', 'amount', 'status', 'description'], deal))
            client_data = self.get_client_by_id(deal_data['client_id'])
            product_data = self.get_product_by_id(deal_data['product_id'])
            company_info = self.get_company_info()

            analysis = self.analytics.predict_deal_success(
                deal_data, client_data, product_data, company_info
            )
            self.save_deal_analysis(deal_id, analysis)
            logging.info(f"Deal analysis regenerated for ID {deal_id}.")
            return analysis
        except Exception as e:
            logging.error(f"Error regenerating deal analysis: {str(e)}")
            st.error(f"Error regenerating deal analysis: {str(e)}")
            return None

    def add_deal(self, client_id: int, product_id: int, amount: float, status: str, description: str = None) -> int:
        try:
            with self.db.conn:
                cursor = self.db.conn.execute('''
                    INSERT INTO deals 
                    (client_id, product_id, amount, status, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (client_id, product_id, amount, status, description))
                deal_id = cursor.lastrowid
                logging.info(f"Deal added with ID: {deal_id}.")
                return deal_id
        except Exception as e:
            logging.error(f"Error adding deal: {str(e)}")
            st.error(f"Error adding deal: {str(e)}")
            return -1

    def update_deal_status(self, deal_id: int, new_status: str) -> bool:
        try:
            with self.db.conn:
                self.db.conn.execute(
                    '''UPDATE deals 
                       SET status = ?, updated_at = CURRENT_TIMESTAMP 
                       WHERE id = ?''',
                    (new_status, deal_id)
                )
                logging.info(f"Deal status updated for ID {deal_id} to {new_status}.")
            return True
        except Exception as e:
            logging.error(f"Error updating deal status: {str(e)}")
            st.error(f"Error updating deal status: {str(e)}")
            return False

    def add_task(self, title: str, description: str, due_date: str, status: str, client_id: Optional[int] = None) -> int:
        try:
            with self.db.conn:
                cursor = self.db.conn.execute(
                    'INSERT INTO tasks (title, description, due_date, status, client_id) VALUES (?, ?, ?, ?, ?)',
                    (title, description, due_date, status, client_id)
                )
                task_id = cursor.lastrowid
                logging.info(f"Task added with ID: {task_id}.")
                return task_id
        except Exception as e:
            logging.error(f"Error adding task: {str(e)}")
            st.error(f"Error adding task: {str(e)}")
            return -1

    def get_client_details(self, client_id: int) -> Optional[Dict]:
        try:
            client = self.get_client_by_id(client_id)
            if not client:
                logging.warning(f"Client details not found for ID {client_id}.")
                return None

            deals = self.db.conn.execute('SELECT * FROM deals WHERE client_id = ?', (client_id,)).fetchall()
            tasks = self.db.conn.execute('SELECT * FROM tasks WHERE client_id = ?', (client_id,)).fetchall()

            logging.info(f"Client details retrieved for ID {client_id}.")
            return {
                'client': client,
                'deals': deals,
                'tasks': tasks
            }
        except Exception as e:
            logging.error(f"Error retrieving client details: {str(e)}")
            st.error(f"Error retrieving client details: {str(e)}")
            return None


def create_streamlit_app():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Simple CRM with AI Analytics", layout="wide")
    st.title("Simple CRM with AI Analytics")

    # Hide Streamlit's default menu and header, and Plotly's modebar
    hide_elements = """
    <style>
    /* Hide Streamlit's default header and menu */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Hide Plotly modebar */
    .modebar {
        display: none !important;
    }
    /* Alternative method to hide Plotly modebar */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    </style>
    """
    st.markdown(hide_elements, unsafe_allow_html=True)

    crm = CRM()

    menu = ["Company Info", "Products/Services", "Clients", "Deals", "Tasks", "Analytics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Company Info":
        st.subheader("Company Information")

        # Add collapsible form
        show_form = st.checkbox("Show Company Info Form", key="show_company_form")

        if show_form:
            with st.form("company_info"):
                # Get existing company info
                company_info = crm.get_company_info()

                name = st.text_input("Company Name", value=company_info.get('name', ''))
                description = st.text_area("Company Description", value=company_info.get('description', ''))
                city = st.text_input("City", value=company_info.get('city', ''))
                industry = st.text_input("Industry", value=company_info.get('industry', ''))
                target_market = st.text_area("Target Market", value=company_info.get('target_market', ''))
                usp = st.text_area("Unique Selling Points", value=company_info.get('unique_selling_points', ''))

                if st.form_submit_button("Save Company Info"):
                    data = {
                        'name': name,
                        'description': description,
                        'city': city,
                        'industry': industry,
                        'target_market': target_market,
                        'unique_selling_points': usp
                    }
                    if crm.save_company_info(data):
                        st.success("Company information saved successfully!")

    elif choice == "Products/Services":
        st.subheader("Products and Services Management")

        # Add collapsible form
        show_form = st.checkbox("Show Add Product/Service Form", key="show_product_form")

        if show_form:
            with st.form("add_product"):
                name = st.text_input("Name")
                type_ = st.selectbox("Type", ["Product", "Service"])
                description = st.text_area("Description")
                price = st.number_input("Price", min_value=0.0, format="%.2f")
                features = st.text_area("Key Features")
                target_audience = st.text_area("Target Audience")

                if st.form_submit_button("Add Product/Service"):
                    data = {
                        'name': name,
                        'type': type_,
                        'description': description,
                        'price': price,
                        'features': features,
                        'target_audience': target_audience
                    }
                    product_id = crm.add_product(data)
                    if product_id != -1:
                        st.success("Product/Service added successfully!")

        # Display products/services
        products = crm.get_products()
        if products:
            st.subheader("Current Products and Services")
            for product in products:
                with st.expander(f"{product['name']} - {product['type']}"):
                    st.write(f"**Description:** {product['description']}")
                    st.write(f"**Price:** ${product['price']:.2f}")
                    st.write(f"**Features:** {product['features']}")
                    st.write(f"**Target Audience:** {product['target_audience']}")

    elif choice == "Clients":
        st.subheader("Client Management")

        # Add collapsible form
        show_form = st.checkbox("Show Add Client Form", key="show_client_form")

        if show_form:
            with st.form("add_client"):
                name = st.text_input("Client Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                company = st.text_input("Client's Company")
                industry = st.text_input("Client's Industry")
                notes = st.text_area("Notes")

                if st.form_submit_button("Add Client"):
                    client_id = crm.add_client(name, email, phone, company, industry, notes)
                    if client_id != -1:
                        st.success("Client added successfully!")

                        # Get company info and products for AI analysis
                        company_info = crm.get_company_info()
                        products = crm.get_products()

                        # Perform AI analysis
                        client_data = {
                            'name': name,
                            'company': company,
                            'industry': industry,
                            'notes': notes
                        }

                        with st.spinner('Analyzing client potential...'):
                            analysis = crm.analytics.analyze_client_potential(
                                client_data, company_info, products
                            )
                            crm.save_client_analysis(client_id, analysis)
                            st.info("Initial AI Analysis saved successfully!")

        clients = pd.read_sql_query("""
            SELECT id, name, email, phone, company, industry, created_at 
            FROM clients
            ORDER BY created_at DESC
        """, crm.db.conn)

        if not clients.empty:
            st.subheader("Client List")
            for _, client in clients.iterrows():
                with st.expander(f"{client['name']} - {client['company']}"):
                    # Basic client info
                    st.write(f"**Email:** {client['email']}")
                    st.write(f"**Phone:** {client['phone']}")
                    st.write(f"**Industry:** {client['industry']}")

                    analysis = crm.get_client_analysis(client['id'])
                    if analysis:
                        # Create columns with space for margin
                        cols = st.columns([1.2, 0.1, 1.2, 3])  # button - space - button - remaining

                        with cols[0]:
                            view_analysis = st.button("📊 View Analysis", key=f"view_client_{client['id']}")

                        with cols[2]:  # Skipping cols[1] for spacing
                            regenerate = st.button("🔄 Regenerate", key=f"regen_client_{client['id']}")

                        # Analysis display
                        if view_analysis:
                            st.info("AI Analysis:")
                            st.markdown(
                                f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                    {analysis['analysis']}
                                </div>""",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Generated at: {analysis['generated_at']}")

                        if regenerate:
                            with st.spinner('Regenerating analysis...'):
                                new_analysis = crm.regenerate_client_analysis(client['id'])
                                st.info("New AI Analysis:")
                                st.markdown(
                                    f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                        {new_analysis}
                                    </div>""",
                                    unsafe_allow_html=True
                                )

    elif choice == "Deals":
        st.subheader("Deal Management")

        # Add collapsible form
        show_form = st.checkbox("Show Add Deal Form", key="show_deal_form")

        if show_form:
            with st.form("add_deal"):
                clients = pd.read_sql_query("SELECT id, name FROM clients", crm.db.conn)
                if clients.empty:
                    st.warning("No clients available. Please add a client first.")
                    client_id = None
                else:
                    client_id = st.selectbox(
                        "Client",
                        clients['id'].tolist(),
                        format_func=lambda x: clients[clients['id'] == x]['name'].iloc[0] if not clients[clients['id'] == x].empty else "Unknown"
                    )

                products = pd.read_sql_query("SELECT id, name, price FROM products", crm.db.conn)
                if products.empty:
                    st.warning("No products/services available. Please add a product/service first.")
                    product_id = None
                else:
                    product_id = st.selectbox(
                        "Product/Service",
                        products['id'].tolist(),
                        format_func=lambda x: products[products['id'] == x]['name'].iloc[0] if not products[products['id'] == x].empty else "Unknown"
                    )

                if product_id and not products[products['id'] == product_id].empty:
                    suggested_price = products[products['id'] == product_id]['price'].iloc[0]
                else:
                    suggested_price = 0.0

                amount = st.number_input("Amount", min_value=0.0, value=suggested_price, format="%.2f")
                status = st.selectbox("Status", crm.DEAL_STATUSES)
                description = st.text_area("Description")

                if st.form_submit_button("Add Deal"):
                    if client_id and product_id:
                        deal_id = crm.add_deal(client_id, product_id, amount, status, description)
                        if deal_id != -1:
                            st.success("Deal added successfully!")

                            # Get data for AI analysis
                            company_info = crm.get_company_info()
                            client_data = crm.get_client_by_id(client_id)
                            product_data = crm.get_product_by_id(product_id)

                            deal_data = {
                                'amount': amount,
                                'status': status,
                                'description': description
                            }

                            with st.spinner('Analyzing deal...'):
                                analysis = crm.analytics.predict_deal_success(
                                    deal_data, client_data, product_data, company_info
                                )
                                crm.save_deal_analysis(deal_id, analysis)
                                st.info("Initial AI Deal Analysis saved successfully!")
                    else:
                        st.error("Please select a valid client and product/service.")

        # Display and manage existing deals
        deals = pd.read_sql_query("""
            SELECT 
                deals.id,
                clients.name as client_name,
                products.name as product_name,
                deals.amount,
                deals.status,
                deals.description,
                deals.created_at,
                deals.updated_at
            FROM deals 
            JOIN clients ON deals.client_id = clients.id
            JOIN products ON deals.product_id = products.id
            ORDER BY deals.updated_at DESC
        """, crm.db.conn)

        if not deals.empty:
            st.subheader("Current Deals")
            for _, deal in deals.iterrows():
                with st.expander(f"Deal #{deal['id']} - {deal['client_name']} - {deal['product_name']}"):
                    st.write(f"**Amount:** ${deal['amount']:,.2f}")
                    st.write(f"**Current Status:** {deal['status']}")
                    st.write(f"**Description:** {deal['description']}")

                    new_status = st.selectbox(
                        "Update Status",
                        crm.DEAL_STATUSES,
                        key=f"status_deal_{deal['id']}",
                        index=crm.DEAL_STATUSES.index(deal['status']) if deal['status'] in crm.DEAL_STATUSES else 0
                    )

                    if new_status != deal['status']:
                        if crm.update_deal_status(deal['id'], new_status):
                            st.success("Deal status updated successfully!")
                            st.experimental_rerun()

                    analysis = crm.get_deal_analysis(deal['id'])
                    if analysis:
                        # Create columns with space for margin
                        cols = st.columns([1.2, 0.1, 1.2, 3])  # button - space - button - remaining

                        with cols[0]:
                            view_analysis = st.button("📊 View Analysis", key=f"view_deal_{deal['id']}")

                        with cols[2]:  # Skipping cols[1] for spacing
                            regenerate = st.button("🔄 Regenerate", key=f"regen_deal_{deal['id']}")

                        # Analysis display
                        if view_analysis:
                            st.info("AI Analysis:")
                            st.markdown(
                                f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                    {analysis['analysis']}
                                </div>""",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Generated at: {analysis['generated_at']}")

                        if regenerate:
                            with st.spinner('Regenerating analysis...'):
                                new_analysis = crm.regenerate_deal_analysis(deal['id'])
                                if new_analysis:
                                    st.info("New AI Analysis:")
                                    st.markdown(
                                        f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                            {new_analysis}
                                        </div>""",
                                        unsafe_allow_html=True
                                    )

            # Deals visualization
            st.subheader("Deal Analytics")

            # Status distribution
            fig1 = px.pie(deals, names='status', values='amount',
                         title='Deal Amount Distribution by Status')
            st.plotly_chart(fig1)

            # Timeline
            fig2 = px.bar(deals, x='created_at', y='amount', color='status',
                          title='Deals by Time and Status')
            st.plotly_chart(fig2)

    elif choice == "Tasks":
        st.subheader("Task Management")

        # Add collapsible form
        show_form = st.checkbox("Show Add Task Form", key="show_task_form")

        if show_form:
            with st.form("add_task"):
                title = st.text_input("Task Title")
                description = st.text_area("Description")
                due_date = st.date_input("Due Date")
                status = st.selectbox("Status", ["New", "In Progress", "Completed"])

                clients = pd.read_sql_query("SELECT id, name FROM clients", crm.db.conn)
                if not clients.empty:
                    client_id = st.selectbox(
                        "Related Client",
                        clients['id'].tolist(),
                        format_func=lambda x: clients[clients['id'] == x]['name'].iloc[0] if not clients[clients['id'] == x].empty else "Unknown"
                    )
                else:
                    st.warning("No clients available. Please add a client first.")
                    client_id = None

                if st.form_submit_button("Add Task"):
                    if client_id:
                        task_id = crm.add_task(
                            title, description, due_date.strftime('%Y-%m-%d'),
                            status, client_id
                        )
                        if task_id != -1:
                            st.success("Task added successfully!")
                    else:
                        st.error("Cannot add task without selecting a valid client.")

        # Display tasks
        tasks = pd.read_sql_query("""
            SELECT tasks.*, clients.name as client_name 
            FROM tasks 
            JOIN clients ON tasks.client_id = clients.id
            ORDER BY due_date ASC
        """, crm.db.conn)

        if not tasks.empty:
            st.subheader("Current Tasks")

            # Group tasks by status
            for status_group in ["New", "In Progress", "Completed"]:
                status_tasks = tasks[tasks['status'] == status_group]
                if not status_tasks.empty:
                    st.write(f"### {status_group}")
                    for _, task in status_tasks.iterrows():
                        with st.expander(f"{task['title']} - Due: {task['due_date']}"):
                            st.write(f"**Description:** {task['description']}")
                            st.write(f"**Client:** {task['client_name']}")
                            st.write(f"**Created:** {task['created_at']}")

    elif choice == "Analytics":
        st.subheader("Business Analytics")

        # Key metrics
        total_clients = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM clients", crm.db.conn)['count'][0]
        total_deals = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM deals", crm.db.conn)['count'][0]
        total_amount = pd.read_sql_query(
                "SELECT SUM(amount) as sum FROM deals", crm.db.conn)['sum'][0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clients", total_clients)
        col2.metric("Total Deals", total_deals)
        col3.metric("Total Deal Amount", f"${total_amount:,.2f}" if total_amount else "$0.00")

        # Sales analytics
        st.subheader("Sales Performance")

        deals_by_status = pd.read_sql_query("""
                SELECT status, COUNT(*) as count, SUM(amount) as total_amount
                FROM deals
                GROUP BY status
        """, crm.db.conn)

        if not deals_by_status.empty:
                col1, col2 = st.columns(2)
            
                with col1:
                        fig1 = px.funnel(
                                deals_by_status,
                                x='count', y='status',
                                title='Sales Funnel by Deal Count'
                        )
                        st.plotly_chart(fig1)
                    
                with col2:
                        fig2 = px.funnel(
                                deals_by_status,
                                x='total_amount', y='status',
                                title='Sales Funnel by Amount'
                        )
                        st.plotly_chart(fig2)
                    
        # Time-based analysis
        st.subheader("Time-Based Analysis")

        deals_timeline = pd.read_sql_query("""
                SELECT 
                        date(created_at) as date,
                        COUNT(*) as deal_count,
                        SUM(amount) as total_amount
                FROM deals
                GROUP BY date(created_at)
                ORDER BY date(created_at)
        """, crm.db.conn)

        if not deals_timeline.empty:
                fig3 = px.line(deals_timeline, x='date', y='deal_count',
                                        title='Number of Deals Over Time', markers=True)
                st.plotly_chart(fig3)
            
                fig4 = px.area(deals_timeline, x='date', y='total_amount',
                                        title='Total Deal Amount Over Time', markers=True)
                st.plotly_chart(fig4)


if __name__ == "__main__":
    create_streamlit_app()
