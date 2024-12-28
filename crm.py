import sqlite3
import datetime
from typing import List, Dict, Optional, Any, Generator, TypeVar, ParamSpec, Callable, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import logging
import time
import queue
import threading
import re
import phonenumbers
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback

# Configure logging with RotatingFileHandler to prevent log files from growing indefinitely
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = RotatingFileHandler('crm_app.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Type hints for decorator
P = ParamSpec('P')
R = TypeVar('R')

# Custom Exceptions with additional attributes
class CRMException(Exception):
    """Base exception class for CRM errors"""
    def __init__(self, message: str, error_code: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class DatabaseError(CRMException):
    """Database related errors"""
    pass

class APIError(CRMException):
    """API related errors"""
    pass

class ValidationError(CRMException):
    """Data validation errors"""
    pass

def with_retry(
    max_attempts: int = 3,
    base_wait: float = 1,
    max_wait: float = 10
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for retrying operations with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        base_wait: Initial wait time between retries
        max_wait: Maximum wait time between retries
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_wait, max=max_wait),
            retry=retry_if_exception_type((requests.RequestException, DatabaseError, APIError)),
            before_sleep=lambda retry_state: logger.warning(
                f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}"
            )
        )
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, (requests.RequestException, DatabaseError, APIError)):
                    raise
                logger.error(f"Unhandled error in {func.__name__}: {str(e)}", exc_info=True)
                raise CRMException(f"Operation failed: {str(e)}")
        return wrapper
    return decorator

class DatabaseConnectionPool:
    def __init__(self, db_name: str, max_connections: int = 5):
        self.db_name = db_name
        self.max_connections = max_connections
        self.connections: queue.Queue = queue.Queue(maxsize=max_connections)
        self.connection_count = 0
        self._lock = threading.Lock()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with pragmas for better performance"""
        conn = sqlite3.connect(
            self.db_name,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        logger.info("Created new database connection")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool"""
        connection = None
        try:
            try:
                connection = self.connections.get_nowait()
                logger.debug("Reusing existing database connection")
            except queue.Empty:
                with self._lock:
                    if self.connection_count < self.max_connections:
                        connection = self._create_connection()
                        self.connection_count += 1
                    else:
                        logger.debug("Waiting for available database connection")
                        connection = self.connections.get()

            yield connection

        except Exception as e:
            logger.error(f"Database connection error: {str(e)}", exc_info=True)
            if connection:
                connection.close()
            raise DatabaseError(f"Database connection failed: {str(e)}")
        finally:
            if connection:
                try:
                    self.connections.put_nowait(connection)
                except queue.Full:
                    connection.close()
                    with self._lock:
                        self.connection_count -= 1
                    logger.debug("Connection pool full. Closed excess connection.")

class Database:
    def __init__(self, db_name: str = "crm.db"):
        self.pool = DatabaseConnectionPool(db_name)
        self.create_tables()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions"""
        with self.pool.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {str(e)}", exc_info=True)
                raise DatabaseError(f"Transaction failed: {str(e)}")

    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a database query"""
        with self.transaction() as conn:
            try:
                cursor = conn.execute(query, params)
                results = cursor.fetchall()
                logger.debug(f"Executed query: {query} | Params: {params}")
                return results
            except sqlite3.Error as e:
                logger.error(f"Query execution failed: {str(e)} | Query: {query} | Params: {params}", exc_info=True)
                raise DatabaseError(f"Query execution failed: {str(e)}")

    def execute_many(self, query: str, params_list: list[tuple]) -> None:
        """Execute multiple database queries"""
        with self.transaction() as conn:
            try:
                conn.executemany(query, params_list)
                logger.debug(f"Executed many queries: {query} | Number of params sets: {len(params_list)}")
            except sqlite3.Error as e:
                logger.error(f"Batch query execution failed: {str(e)} | Query: {query}", exc_info=True)
                raise DatabaseError(f"Batch query execution failed: {str(e)}")

    def create_tables(self):
        """Create all necessary database tables"""
        tables = [
            """
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
            """,
            """
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
            """,
            """
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
            """,
            """
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
            """,
            """
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
            """,
            """
            CREATE TABLE IF NOT EXISTS client_analyses (
                id INTEGER PRIMARY KEY,
                client_id INTEGER,
                analysis TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES clients (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS deal_analyses (
                id INTEGER PRIMARY KEY,
                deal_id INTEGER,
                analysis TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (deal_id) REFERENCES deals (id)
            )
            """
        ]

        try:
            with self.transaction() as conn:
                for table in tables:
                    conn.execute(table)
                logger.info("Database tables created successfully or already exist.")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error creating tables: {str(e)}")

class InputValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str, region: str = "US") -> bool:
        """Validate phone number format"""
        try:
            number = phonenumbers.parse(phone, region)
            return phonenumbers.is_valid_number(number)
        except phonenumbers.NumberParseException:
            return False

    @staticmethod
    def validate_amount(amount: float) -> bool:
        """Validate monetary amount"""
        return 0 <= amount <= 1e9  # Arbitrary upper limit

    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input"""
        # Remove any potential SQL injection or XSS attempts
        text = re.sub(r'[<>]', '', text)
        text = text.replace("'", "''")
        return text[:max_length]

    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date format (YYYY-MM-DD)"""
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

class DataValidator:
    def validate_company_info(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Validate company information"""
        errors = {}

        if not data.get('name'):
            errors['name'] = "Company name is required"
        elif len(data['name']) > 100:
            errors['name'] = "Company name is too long"

        if len(data.get('description', '')) > 1000:
            errors['description'] = "Description is too long"

        return errors

    def validate_product(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Validate product information"""
        errors = {}

        if not data.get('name'):
            errors['name'] = "Product name is required"

        if data.get('type') not in ['Product', 'Service']:
            errors['type'] = "Invalid product type"

        if not InputValidator.validate_amount(data.get('price', 0)):
            errors['price'] = "Invalid price"

        return errors

    def validate_client(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Validate client information"""
        errors = {}

        if not data.get('name'):
            errors['name'] = "Client name is required"

        if data.get('email') and not InputValidator.validate_email(data['email']):
            errors['email'] = "Invalid email format"

        if data.get('phone') and not InputValidator.validate_phone(data['phone']):
            errors['phone'] = "Invalid phone number format"

        return errors

    def validate_deal(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Validate deal information"""
        errors = {}

        if not data.get('client_id'):
            errors['client_id'] = "Client is required"

        if not data.get('product_id'):
            errors['product_id'] = "Product is required"

        if not InputValidator.validate_amount(data.get('amount', 0)):
            errors['amount'] = "Invalid amount"

        if data.get('status') not in ["New", "In Progress", "Won", "Lost", "On Hold"]:
            errors['status'] = "Invalid deal status"

        return errors

@dataclass
class RateLimitRule:
    requests: int
    period: int  # in seconds

class InMemoryRateLimiter:
    def __init__(self):
        self._requests = {}  # Dictionary to store request timestamps
        self._lock = threading.Lock()

        # Define rate limit rules
        self.rules = {
            'default': RateLimitRule(100, 3600),       # 100 requests per hour
            'ai_analysis': RateLimitRule(10, 60),     # 10 AI analysis requests per minute
            'database': RateLimitRule(1000, 3600),    # 1000 database operations per hour
        }

    def _clean_old_requests(self, key: str, rule: RateLimitRule) -> None:
        """Remove requests older than the time window"""
        current_time = time.time()
        self._requests[key] = [
            timestamp for timestamp in self._requests.get(key, [])
            if current_time - timestamp < rule.period
        ]

    def check_rate_limit(self, key: str, rule_name: str = 'default') -> Tuple[bool, Optional[float]]:
        """
        Check if the request should be rate limited

        Returns:
            Tuple[bool, Optional[float]]: (is_allowed, retry_after)
        """
        with self._lock:
            rule = self.rules.get(rule_name, self.rules['default'])

            if key not in self._requests:
                self._requests[key] = []

            self._clean_old_requests(key, rule)

            if len(self._requests[key]) >= rule.requests:
                oldest_timestamp = min(self._requests[key])
                retry_after = rule.period - (time.time() - oldest_timestamp)
                return False, max(0, retry_after)

            self._requests[key].append(time.time())
            return True, None

class RateLimitedAPI:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.rate_limiter = InMemoryRateLimiter()
        self.ollama_url = ollama_url

    @with_retry(max_attempts=3)
    def make_ai_request(self, key: str, prompt: str) -> str:
        """Make an AI analysis request with rate limiting"""
        is_allowed, retry_after = self.rate_limiter.check_rate_limit(key, 'ai_analysis')

        if not is_allowed:
            raise APIError(f"Rate limit exceeded. Try again in {retry_after:.1f} seconds")

        try:
            payload = {
                "model": "gemma2:9b",
                "prompt": prompt,
                "stream": False
            }
            logger.debug(f"Sending AI request for {key} with payload: {json.dumps(payload)}")

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30  # Increased timeout to 30 seconds
            )
            response.raise_for_status()
            response_json = response.json()
            logger.debug(f"AI Response for {key}: {response_json}")
            return response_json.get('response', "No response from AI model.")

        except requests.RequestException as e:
            logger.error(f"AI request failed: {str(e)}", exc_info=True)
            logger.error(f"Response Content: {response.text if 'response' in locals() else 'No response'}")
            raise APIError(f"AI request failed: {str(e)}")

def serialize_for_json(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    else:
        return obj

class CRMAnalytics:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.api = RateLimitedAPI(ollama_url)

    def analyze_client_potential(self, client_data: Dict, company_info: Dict, products: List[Dict]) -> str:
        """Analyze client potential using AI"""
        # Serialize all data before sending
        client_data_serialized = serialize_for_json(client_data)
        company_info_serialized = serialize_for_json(company_info)
        products_serialized = serialize_for_json(products)

        prompt = f"""
        Based on the following information, analyze the client's potential:

        Company Context:
        - Company: {company_info_serialized.get('name')}
        - Industry: {company_info_serialized.get('industry')}
        - City: {company_info_serialized.get('city')}
        - Target Market: {company_info_serialized.get('target_market')}
        - Unique Selling Points: {company_info_serialized.get('unique_selling_points')}

        Client Information:
        - Name: {client_data_serialized.get('name')}
        - Industry: {client_data_serialized.get('industry')}
        - Company: {client_data_serialized.get('company')}
        - Notes: {client_data_serialized.get('notes')}

        Available Products/Services:
        {json.dumps(products_serialized, indent=2)}

        Please provide a comprehensive analysis:
        1. Client-Product Fit
        2. Probability of successful deals
        3. Recommended products/services for this client
        4. Suggested approach and next actions
        5. Potential deal value estimation
        """
        return self.api.make_ai_request(f"client_analysis_{client_data_serialized.get('id')}", prompt)

    def predict_deal_success(self, deal_data: Dict, client_data: Dict, product_data: Dict, company_info: Dict) -> str:
        """Predict deal success probability using AI"""
        # Serialize all data before sending
        deal_data_serialized = serialize_for_json(deal_data)
        client_data_serialized = serialize_for_json(client_data)
        product_data_serialized = serialize_for_json(product_data)
        company_info_serialized = serialize_for_json(company_info)

        prompt = f"""
        Analyze the probability of successfully closing this deal based on the following context:

        Company Information:
        - Company: {company_info_serialized.get('name')}
        - Industry: {company_info_serialized.get('industry')}
        - City: {company_info_serialized.get('city')}
        - Target Market: {company_info_serialized.get('target_market')}
        - Unique Selling Points: {company_info_serialized.get('unique_selling_points')}

        Client Information:
        - Name: {client_data_serialized.get('name')}
        - Industry: {client_data_serialized.get('industry')}
        - Company: {client_data_serialized.get('company')}

        Product/Service Information:
        - Name: {product_data_serialized.get('name')}
        - Type: {product_data_serialized.get('type')}
        - Price: {product_data_serialized.get('price')}
        - Target Audience: {product_data_serialized.get('target_audience')}

        Deal Information:
        - Amount: {deal_data_serialized.get('amount')}
        - Description: {deal_data_serialized.get('description')}

        Please provide:
        1. Success probability percentage
        2. Key risk factors
        3. Competitive advantages
        4. Recommended negotiation strategy
        5. Value proposition alignment
        6. Specific actions to increase success chances
        """
        return self.api.make_ai_request(f"deal_analysis_{deal_data_serialized.get('id')}", prompt)

class CRM:
    def __init__(self):
        self.db = Database()
        self.validator = DataValidator()
        self.analytics = CRMAnalytics()
        self.DEAL_STATUSES = ["New", "In Progress", "Won", "Lost", "On Hold"]

    def save_company_info(self, data: Dict) -> bool:
        """Save company information"""
        try:
            # Validate input data
            errors = self.validator.validate_company_info(data)
            if errors:
                raise ValidationError(str(errors))

            # Sanitize text inputs
            data = {k: InputValidator.sanitize_text(str(v)) if v else v 
                   for k, v in data.items()}

            with self.db.transaction() as conn:
                # Check if company info exists
                existing = conn.execute('SELECT id FROM company_info LIMIT 1').fetchone()

                if existing:
                    # Update
                    conn.execute('''
                        UPDATE company_info 
                        SET name=?, description=?, city=?, industry=?, target_market=?, 
                            unique_selling_points=?, updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    ''', (
                        data['name'], data['description'], data['city'], data['industry'],
                        data['target_market'], data['unique_selling_points'], existing[0]
                    ))
                    logger.info("Company information updated.")
                else:
                    # Insert
                    conn.execute('''
                        INSERT INTO company_info 
                        (name, description, city, industry, target_market, unique_selling_points)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        data['name'], data['description'], data['city'], data['industry'],
                        data['target_market'], data['unique_selling_points']
                    ))
                    logger.info("Company information added.")
            return True
        except Exception as e:
            logger.error(f"Error saving company information: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error saving company information: {str(e)}")

    def get_company_info(self) -> Dict:
        """Get company information"""
        try:
            result = self.db.execute('''
                SELECT name, description, city, industry, target_market, unique_selling_points
                FROM company_info
                LIMIT 1
            ''')

            if result:
                company_info = dict(zip(
                    ['name', 'description', 'city', 'industry', 'target_market', 'unique_selling_points'],
                    result[0]
                ))
                logger.info("Company information retrieved.")
                return company_info
            logger.info("No company information found.")
            return {}
        except Exception as e:
            logger.error(f"Error retrieving company information: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving company information: {str(e)}")

    def add_product(self, data: Dict) -> int:
        """Add a new product"""
        try:
            # Validate input data
            errors = self.validator.validate_product(data)
            if errors:
                raise ValidationError(str(errors))

            # Sanitize text inputs
            data = {k: InputValidator.sanitize_text(str(v)) if v else v 
                   for k, v in data.items()}

            with self.db.transaction() as conn:
                cursor = conn.execute('''
                    INSERT INTO products 
                    (name, type, description, price, features, target_audience)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['name'], data['type'], data['description'],
                    data['price'], data['features'], data['target_audience']
                ))
                product_id = cursor.lastrowid
                logger.info(f"Product/Service added with ID: {product_id}.")
                return product_id
        except Exception as e:
            logger.error(f"Error adding product/service: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error adding product/service: {str(e)}")

    def get_products(self) -> List[Dict]:
        """Get all products"""
        try:
            results = self.db.execute('SELECT * FROM products')
            columns = ['id', 'name', 'type', 'description', 'price', 'features', 'target_audience', 'created_at']
            products = [dict(zip(columns, row)) for row in results]
            logger.info(f"{len(products)} products/services retrieved.")
            return products
        except Exception as e:
            logger.error(f"Error retrieving products/services: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving products/services: {str(e)}")

    def get_product_by_id(self, product_id: int) -> Dict:
        """Get product by ID"""
        try:
            result = self.db.execute('SELECT * FROM products WHERE id = ?', (product_id,))
            if not result:
                logger.warning(f"Product with ID {product_id} not found.")
                return {}

            columns = ['id', 'name', 'type', 'description', 'price', 'features', 'target_audience', 'created_at']
            product = dict(zip(columns, result[0]))
            logger.info(f"Product retrieved by ID: {product_id}.")
            return product
        except Exception as e:
            logger.error(f"Error retrieving product by ID {product_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving product by ID {product_id}: {str(e)}")

    def add_client(self, data: Dict) -> int:
        """Add a new client"""
        try:
            # Validate input data
            errors = self.validator.validate_client(data)
            if errors:
                raise ValidationError(str(errors))

            # Sanitize text inputs
            data = {k: InputValidator.sanitize_text(str(v)) if v else v 
                   for k, v in data.items()}

            with self.db.transaction() as conn:
                cursor = conn.execute('''
                    INSERT INTO clients 
                    (name, email, phone, company, industry, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    data['name'], data.get('email'), data.get('phone'),
                    data.get('company'), data.get('industry'), data.get('notes')
                ))
                client_id = cursor.lastrowid
                logger.info(f"Client added with ID: {client_id}.")
                return client_id
        except Exception as e:
            logger.error(f"Error adding client: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error adding client: {str(e)}")

    def get_client_by_id(self, client_id: int) -> Dict:
        """Get client by ID"""
        try:
            result = self.db.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
            if not result:
                logger.warning(f"Client with ID {client_id} not found.")
                return {}

            columns = ['id', 'name', 'email', 'phone', 'company', 'industry', 'notes', 'created_at']
            client = dict(zip(columns, result[0]))
            logger.info(f"Client retrieved by ID: {client_id}.")
            return client
        except Exception as e:
            logger.error(f"Error retrieving client by ID {client_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving client by ID {client_id}: {str(e)}")

    def save_client_analysis(self, client_id: int, analysis: str) -> bool:
        """Save client analysis"""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT INTO client_analyses (client_id, analysis)
                    VALUES (?, ?)
                ''', (client_id, InputValidator.sanitize_text(analysis)))
                logger.info(f"Client analysis saved for ID {client_id}.")
            return True
        except Exception as e:
            logger.error(f"Error saving client analysis: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error saving client analysis: {str(e)}")

    def save_deal_analysis(self, deal_id: int, analysis: str) -> bool:
        """Save deal analysis"""
        try:
            with self.db.transaction() as conn:
                conn.execute('''
                    INSERT INTO deal_analyses (deal_id, analysis)
                    VALUES (?, ?)
                ''', (deal_id, InputValidator.sanitize_text(analysis)))
                logger.info(f"Deal analysis saved for ID {deal_id}.")
            return True
        except Exception as e:
            logger.error(f"Error saving deal analysis: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error saving deal analysis: {str(e)}")

    def get_client_analysis(self, client_id: int) -> Optional[Dict]:
        """Get latest client analysis"""
        try:
            result = self.db.execute('''
                SELECT analysis, generated_at 
                FROM client_analyses 
                WHERE client_id = ? 
                ORDER BY generated_at DESC 
                LIMIT 1
            ''', (client_id,))

            if result:
                analysis = {
                    'analysis': result[0][0],
                    'generated_at': result[0][1]
                }
                logger.info(f"Client analysis retrieved for ID {client_id}.")
                return analysis
            logger.info(f"No client analysis found for ID {client_id}.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving client analysis: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving client analysis: {str(e)}")

    def get_deal_analysis(self, deal_id: int) -> Optional[Dict]:
        """Get latest deal analysis"""
        try:
            result = self.db.execute('''
                SELECT analysis, generated_at 
                FROM deal_analyses 
                WHERE deal_id = ? 
                ORDER BY generated_at DESC 
                LIMIT 1
            ''', (deal_id,))

            if result:
                analysis = {
                    'analysis': result[0][0],
                    'generated_at': result[0][1]
                }
                logger.info(f"Deal analysis retrieved for ID {deal_id}.")
                return analysis
            logger.info(f"No deal analysis found for ID {deal_id}.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving deal analysis: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving deal analysis: {str(e)}")

    def regenerate_client_analysis(self, client_id: int) -> str:
        """Regenerate client analysis using AI"""
        try:
            client_data = self.get_client_by_id(client_id)
            if not client_data:
                raise ValidationError(f"Client with ID {client_id} not found")

            company_info = self.get_company_info()
            products = self.get_products()

            logger.debug(f"Client Data: {client_data}")
            logger.debug(f"Company Info: {company_info}")
            logger.debug(f"Products: {products}")

            analysis = self.analytics.analyze_client_potential(
                client_data, company_info, products
            )
            logger.debug(f"AI Analysis Result: {analysis}")
            self.save_client_analysis(client_id, analysis)
            logger.info(f"Client analysis regenerated for ID {client_id}.")
            return analysis
        except APIError as e:
            logger.error(f"APIError during client analysis regeneration: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error regenerating client analysis: {str(e)}", exc_info=True)
            raise APIError(f"Error regenerating client analysis: {str(e)}")

    def regenerate_deal_analysis(self, deal_id: int) -> str:
        """Regenerate deal analysis using AI"""
        try:
            deal_result = self.db.execute('SELECT * FROM deals WHERE id = ?', (deal_id,))
            if not deal_result:
                raise ValidationError(f"Deal with ID {deal_id} not found")

            columns = ['id', 'client_id', 'product_id', 'amount', 'status', 'description', 'created_at', 'updated_at']
            deal_data = dict(zip(columns, deal_result[0]))

            client_data = self.get_client_by_id(deal_data['client_id'])
            product_data = self.get_product_by_id(deal_data['product_id'])
            company_info = self.get_company_info()

            analysis = self.analytics.predict_deal_success(
                deal_data, client_data, product_data, company_info
            )
            self.save_deal_analysis(deal_id, analysis)
            logger.info(f"Deal analysis regenerated for ID {deal_id}.")
            return analysis
        except APIError as e:
            logger.error(f"APIError during deal analysis regeneration: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error regenerating deal analysis: {str(e)}", exc_info=True)
            raise APIError(f"Error regenerating deal analysis: {str(e)}")

    def add_deal(self, data: Dict) -> int:
        """Add a new deal"""
        try:
            # Validate input data
            errors = self.validator.validate_deal(data)
            if errors:
                raise ValidationError(str(errors))

            # Sanitize text inputs
            data = {k: InputValidator.sanitize_text(str(v)) if isinstance(v, str) else v 
                   for k, v in data.items()}

            with self.db.transaction() as conn:
                cursor = conn.execute('''
                    INSERT INTO deals 
                    (client_id, product_id, amount, status, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    data['client_id'], data['product_id'], data['amount'],
                    data['status'], data.get('description')
                ))
                deal_id = cursor.lastrowid
                logger.info(f"Deal added with ID: {deal_id}.")
                return deal_id
        except Exception as e:
            logger.error(f"Error adding deal: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error adding deal: {str(e)}")

    def update_deal_status(self, deal_id: int, new_status: str) -> bool:
        """Update deal status"""
        try:
            if new_status not in self.DEAL_STATUSES:
                raise ValidationError(f"Invalid deal status: {new_status}")

            with self.db.transaction() as conn:
                conn.execute(
                    '''UPDATE deals 
                       SET status = ?, updated_at = CURRENT_TIMESTAMP 
                       WHERE id = ?''',
                    (new_status, deal_id)
                )
                logger.info(f"Deal status updated for ID {deal_id} to {new_status}.")
            return True
        except Exception as e:
            logger.error(f"Error updating deal status: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error updating deal status: {str(e)}")

    def add_task(self, data: Dict) -> int:
        """Add a new task"""
        try:
            # Validate date format
            if not InputValidator.validate_date(data['due_date']):
                raise ValidationError("Invalid date format")

            # Sanitize text inputs
            data = {k: InputValidator.sanitize_text(str(v)) if isinstance(v, str) else v 
                   for k, v in data.items()}

            with self.db.transaction() as conn:
                cursor = conn.execute(
                    'INSERT INTO tasks (title, description, due_date, status, client_id) VALUES (?, ?, ?, ?, ?)',
                    (data['title'], data['description'], data['due_date'], data['status'], data.get('client_id'))
                )
                task_id = cursor.lastrowid
                logger.info(f"Task added with ID: {task_id}.")
                return task_id
        except Exception as e:
            logger.error(f"Error adding task: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error adding task: {str(e)}")

    def get_client_details(self, client_id: int) -> Optional[Dict]:
        """Get comprehensive client details including deals and tasks"""
        try:
            client = self.get_client_by_id(client_id)
            if not client:
                logger.warning(f"Client details not found for ID {client_id}.")
                return None

            deals = self.db.execute(
                'SELECT * FROM deals WHERE client_id = ?', 
                (client_id,)
            )
            deal_columns = ['id', 'client_id', 'product_id', 'amount', 'status', 'description', 'created_at', 'updated_at']

            tasks = self.db.execute(
                'SELECT * FROM tasks WHERE client_id = ?', 
                (client_id,)
            )
            task_columns = ['id', 'title', 'description', 'due_date', 'status', 'client_id', 'created_at']

            client_details = {
                'client': client,
                'deals': [dict(zip(deal_columns, deal)) for deal in deals],
                'tasks': [dict(zip(task_columns, task)) for task in tasks]
            }

            logger.info(f"Client details retrieved for ID {client_id}.")
            return client_details
        except Exception as e:
            logger.error(f"Error retrieving client details: {str(e)}", exc_info=True)
            raise DatabaseError(f"Error retrieving client details: {str(e)}")

def create_streamlit_app():
    """Create and configure the Streamlit application"""
    # Set Streamlit page configuration
    st.set_page_config(page_title="Enhanced CRM with AI Analytics", layout="wide")
    st.title("Enhanced CRM with AI Analytics")

    # Hide Streamlit's default menu and header
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

    try:
        crm = CRM()

        menu = ["Company Info", "Products/Services", "Clients", "Deals", "Tasks", "Analytics"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Company Info":
            st.subheader("Company Information")
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
                        try:
                            data = {
                                'name': name,
                                'description': description,
                                'city': city,
                                'industry': industry,
                                'target_market': target_market,
                                'unique_selling_points': usp
                            }
                            crm.save_company_info(data)
                            st.success("Company information saved successfully!")
                        except ValidationError as e:
                            st.error(f"Validation error: {str(e)}")
                        except DatabaseError as e:
                            st.error(f"Database error: {str(e)}")
                        except CRMException as e:
                            st.error(f"Unexpected error: {str(e)}")

        elif choice == "Products/Services":
            st.subheader("Products and Services Management")
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
                        try:
                            data = {
                                'name': name,
                                'type': type_,
                                'description': description,
                                'price': price,
                                'features': features,
                                'target_audience': target_audience
                            }
                            product_id = crm.add_product(data)
                            st.success("Product/Service added successfully!")
                        except ValidationError as e:
                            st.error(f"Validation error: {str(e)}")
                        except DatabaseError as e:
                            st.error(f"Database error: {str(e)}")
                        except CRMException as e:
                            st.error(f"Unexpected error: {str(e)}")

            # Display products/services
            try:
                products = crm.get_products()
                if products:
                    st.subheader("Current Products and Services")
                    for product in products:
                        with st.expander(f"{product['name']} - {product['type']}"):
                            st.write(f"**Description:** {product['description']}")
                            st.write(f"**Price:** ${product['price']:.2f}")
                            st.write(f"**Features:** {product['features']}")
                            st.write(f"**Target Audience:** {product['target_audience']}")
            except DatabaseError as e:
                st.error(f"Error loading products: {str(e)}")
            except CRMException as e:
                st.error(f"Unexpected error: {str(e)}")

        elif choice == "Clients":
            st.subheader("Client Management")
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
                        try:
                            data = {
                                'name': name,
                                'email': email,
                                'phone': phone,
                                'company': company,
                                'industry': industry,
                                'notes': notes
                            }
                            client_id = crm.add_client(data)
                            st.success("Client added successfully!")

                            # Generate initial AI analysis
                            with st.spinner('Analyzing client potential...'):
                                analysis = crm.regenerate_client_analysis(client_id)
                                st.info("Initial AI Analysis saved successfully!")
                        except ValidationError as e:
                            st.error(f"Validation error: {str(e)}")
                        except DatabaseError as e:
                            st.error(f"Database error: {str(e)}")
                        except APIError as e:
                            st.error(f"API error: {str(e)}")
                        except CRMException as e:
                            st.error(f"Unexpected error: {str(e)}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            st.error(traceback.format_exc())

            # Display clients
            try:
                with crm.db.pool.get_connection() as conn:
                    clients = pd.read_sql_query("""
                        SELECT id, name, email, phone, company, industry, created_at 
                        FROM clients
                        ORDER BY created_at DESC
                    """, conn)

                if not clients.empty:
                    st.subheader("Client List")
                    for _, client in clients.iterrows():
                        with st.expander(f"{client['name']} - {client['company']}"):
                            st.write(f"**Email:** {client['email']}")
                            st.write(f"**Phone:** {client['phone']}")
                            st.write(f"**Industry:** {client['industry']}")

                            analysis = crm.get_client_analysis(client['id'])
                            if analysis:
                                cols = st.columns([1.2, 0.1, 1.2, 3])

                                with cols[0]:
                                    view_analysis = st.button("📊 View Analysis", key=f"view_client_{client['id']}")

                                with cols[2]:
                                    regenerate = st.button("🔄 Regenerate", key=f"regen_client_{client['id']}")

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
                                    try:
                                        with st.spinner('Regenerating analysis...'):
                                            new_analysis = crm.regenerate_client_analysis(client['id'])
                                            st.info("New AI Analysis:")
                                            st.markdown(
                                                f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                                    {new_analysis}
                                                </div>""",
                                                unsafe_allow_html=True
                                            )
                                    except APIError as e:
                                        st.error(f"API error: {str(e)}")
                                    except CRMException as e:
                                        st.error(f"Unexpected error: {str(e)}")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred: {str(e)}")
                                        st.error(traceback.format_exc())
            except DatabaseError as e:
                st.error(f"Error loading clients: {str(e)}")
            except CRMException as e:
                st.error(f"Unexpected error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error(traceback.format_exc())

        elif choice == "Deals":
            st.subheader("Deal Management")
            show_form = st.checkbox("Show Add Deal Form", key="show_deal_form")

            if show_form:
                with st.form("add_deal"):
                    try:
                        with crm.db.pool.get_connection() as deals_conn:
                            clients = pd.read_sql_query(
                                "SELECT id, name FROM clients", 
                                deals_conn
                            )
                        if clients.empty:
                            st.warning("No clients available. Please add a client first.")
                            client_id = None
                        else:
                            client_id = st.selectbox(
                                "Client",
                                clients['id'].tolist(),
                                format_func=lambda x: clients[clients['id'] == x]['name'].iloc[0]
                            )

                        with crm.db.pool.get_connection() as deals_conn:
                            products = pd.read_sql_query(
                                "SELECT id, name, price FROM products", 
                                deals_conn
                            )
                        if products.empty:
                            st.warning("No products/services available. Please add a product/service first.")
                            product_id = None
                        else:
                            product_id = st.selectbox(
                                "Product/Service",
                                products['id'].tolist(),
                                format_func=lambda x: products[products['id'] == x]['name'].iloc[0]
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
                                try:
                                    data = {
                                        'client_id': client_id,
                                        'product_id': product_id,
                                        'amount': amount,
                                        'status': status,
                                        'description': description
                                    }
                                    deal_id = crm.add_deal(data)
                                    st.success("Deal added successfully!")

                                    with st.spinner('Analyzing deal...'):
                                        analysis = crm.regenerate_deal_analysis(deal_id)
                                        st.info("Deal analysis saved successfully!")
                                except ValidationError as e:
                                    st.error(f"Validation error: {str(e)}")
                                except DatabaseError as e:
                                    st.error(f"Database error: {str(e)}")
                                except APIError as e:
                                    st.error(f"API error: {str(e)}")
                                except CRMException as e:
                                    st.error(f"Unexpected error: {str(e)}")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred: {str(e)}")
                                    st.error(traceback.format_exc())
                            else:
                                st.error("Client and Product/Service must be selected.")
                    except Exception as e:
                        st.error(f"Error preparing deal form: {str(e)}")
                        st.error(traceback.format_exc())

            # Display deals
            try:
                with crm.db.pool.get_connection() as conn:
                    deals = pd.read_sql_query("""
                        SELECT deals.id, clients.name AS client_name, products.name AS product_name, deals.amount, deals.status, deals.description, deals.created_at, deals.updated_at
                        FROM deals
                        LEFT JOIN clients ON deals.client_id = clients.id
                        LEFT JOIN products ON deals.product_id = products.id
                        ORDER BY deals.created_at DESC
                    """, conn)

                if not deals.empty:
                    st.subheader("Deal List")
                    for _, deal in deals.iterrows():
                        with st.expander(f"Deal ID: {deal['id']} - {deal['client_name']} - {deal['product_name']}"):
                            st.write(f"**Amount:** ${deal['amount']:.2f}")
                            st.write(f"**Status:** {deal['status']}")
                            st.write(f"**Description:** {deal['description']}")
                            st.write(f"**Created At:** {deal['created_at']}")
                            st.write(f"**Updated At:** {deal['updated_at']}")

                            analysis = crm.get_deal_analysis(deal['id'])
                            if analysis:
                                cols = st.columns([1.2, 0.1, 1.2, 3])

                                with cols[0]:
                                    view_analysis = st.button("📊 View Analysis", key=f"view_deal_{deal['id']}")

                                with cols[2]:
                                    regenerate = st.button("🔄 Regenerate", key=f"regen_deal_{deal['id']}")

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
                                    try:
                                        with st.spinner('Regenerating analysis...'):
                                            new_analysis = crm.regenerate_deal_analysis(deal['id'])
                                            st.info("New AI Analysis:")
                                            st.markdown(
                                                f"""<div style='width: 100%; white-space: normal; word-wrap: break-word;'>
                                                    {new_analysis}
                                                </div>""",
                                                unsafe_allow_html=True
                                            )
                                    except APIError as e:
                                        st.error(f"API error: {str(e)}")
                                    except CRMException as e:
                                        st.error(f"Unexpected error: {str(e)}")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred: {str(e)}")
                                        st.error(traceback.format_exc())
            except DatabaseError as e:
                st.error(f"Error loading deals: {str(e)}")
            except CRMException as e:
                st.error(f"Unexpected error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error(traceback.format_exc())

        elif choice == "Tasks":
            st.subheader("Task Management")
            show_form = st.checkbox("Show Add Task Form", key="show_task_form")

            if show_form:
                with st.form("add_task"):
                    title = st.text_input("Title")
                    description = st.text_area("Description")
                    due_date = st.date_input("Due Date")
                    status = st.selectbox("Status", ["Pending", "Completed", "Overdue"])

                    try:
                        with crm.db.pool.get_connection() as conn:
                            clients = pd.read_sql_query(
                                "SELECT id, name FROM clients", 
                                conn
                            )
                        if clients.empty:
                            st.warning("No clients available. Please add a client first.")
                            client_id = None
                        else:
                            client_id = st.selectbox(
                                "Client",
                                clients['id'].tolist(),
                                format_func=lambda x: clients[clients['id'] == x]['name'].iloc[0]
                            )
                    except Exception as e:
                        st.error(f"Error loading clients for tasks: {str(e)}")
                        client_id = None

                    if st.form_submit_button("Add Task"):
                        try:
                            data = {
                                'title': title,
                                'description': description,
                                'due_date': due_date.strftime('%Y-%m-%d'),
                                'status': status,
                                'client_id': client_id
                            }
                            task_id = crm.add_task(data)
                            st.success("Task added successfully!")
                        except ValidationError as e:
                            st.error(f"Validation error: {str(e)}")
                        except DatabaseError as e:
                            st.error(f"Database error: {str(e)}")
                        except CRMException as e:
                            st.error(f"Unexpected error: {str(e)}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            st.error(traceback.format_exc())

            # Display tasks
            try:
                with crm.db.pool.get_connection() as conn:
                    tasks = pd.read_sql_query("""
                        SELECT tasks.id, tasks.title, tasks.description, tasks.due_date, tasks.status, clients.name AS client_name, tasks.created_at
                        FROM tasks
                        LEFT JOIN clients ON tasks.client_id = clients.id
                        ORDER BY tasks.created_at DESC
                    """, conn)

                if not tasks.empty:
                    st.subheader("Task List")
                    for _, task in tasks.iterrows():
                        with st.expander(f"Task ID: {task['id']} - {task['title']}"):
                            st.write(f"**Description:** {task['description']}")
                            st.write(f"**Due Date:** {task['due_date']}")
                            st.write(f"**Status:** {task['status']}")
                            st.write(f"**Client:** {task['client_name']}")
                            st.write(f"**Created At:** {task['created_at']}")
            except DatabaseError as e:
                st.error(f"Error loading tasks: {str(e)}")
            except CRMException as e:
                st.error(f"Unexpected error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error(traceback.format_exc())

        elif choice == "Analytics":
            st.subheader("Analytics Dashboard")
            # Analytics Visualizations
            try:
                # Example: Deals by Status
                deals = crm.db.execute('SELECT status, COUNT(*) FROM deals GROUP BY status')
                if deals:
                    df_deals_status = pd.DataFrame(deals, columns=['Status', 'Count'])
                    fig1 = px.pie(df_deals_status, names='Status', values='Count', title='Deals by Status')
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No deals data available for visualization.")

                # Example: Sales Over Time
                sales = crm.db.execute('SELECT DATE(created_at) as date, SUM(amount) as total FROM deals GROUP BY DATE(created_at) ORDER BY DATE(created_at)')
                if sales:
                    df_sales = pd.DataFrame(sales, columns=['Date', 'Total Sales'])
                    fig2 = px.line(df_sales, x='Date', y='Total Sales', title='Sales Over Time')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No sales data available for visualization.")

                # Example: Top Clients by Deal Amount
                top_clients = crm.db.execute('''
                    SELECT clients.name, SUM(deals.amount) as total_amount
                    FROM deals
                    LEFT JOIN clients ON deals.client_id = clients.id
                    GROUP BY deals.client_id
                    ORDER BY total_amount DESC
                    LIMIT 10
                ''')
                if top_clients:
                    df_top_clients = pd.DataFrame(top_clients, columns=['Client', 'Total Amount'])
                    fig3 = px.bar(df_top_clients, x='Client', y='Total Amount', title='Top Clients by Deal Amount')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No client data available for visualization.")

                # Example: Product Performance
                product_performance = crm.db.execute('''
                    SELECT products.name, COUNT(deals.id) as deal_count, SUM(deals.amount) as total_amount
                    FROM deals
                    LEFT JOIN products ON deals.product_id = products.id
                    GROUP BY deals.product_id
                    ORDER BY total_amount DESC
                    LIMIT 10
                ''')
                if product_performance:
                    df_product_perf = pd.DataFrame(product_performance, columns=['Product', 'Deal Count', 'Total Amount'])
                    fig4 = px.bar(df_product_perf, x='Product', y='Total Amount', title='Top Products by Total Deal Amount')
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No product data available for visualization.")

                # Additional analytics can be added here
            except DatabaseError as e:
                st.error(f"Database error generating analytics: {str(e)}")
            except CRMException as e:
                st.error(f"Unexpected error generating analytics: {str(e)}")
            except Exception as e:
                logger.error(f"Error generating analytics: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred while generating analytics: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error initializing CRM application: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()
