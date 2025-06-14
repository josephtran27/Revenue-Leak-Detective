from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from faker import Faker
import random
import datetime

# Set up database connection and fake data generator
Base = declarative_base()
fake = Faker()

# Connect to PostgreSQL database for Revenue Leak Detective project
engine = create_engine("postgresql+psycopg2://localhost/revenue_db")
Session = sessionmaker(bind=engine)
session = Session()

# Define database tables for e-commerce business simulation

class Customer(Base):
    """Customer profiles with tiers and lifetime value tracking"""
    __tablename__ = 'customers'
    customer_id = Column(Integer, primary_key=True)
    email = Column(String)
    signup_date = Column(Date)
    tier = Column(String)  # bronze, silver, gold for segmentation
    country = Column(String)
    total_orders = Column(Integer)
    lifetime_value = Column(Float)  # Total revenue from customer

class Product(Base):
    """Product catalog with cost and margin calculations"""
    __tablename__ = 'products'
    product_id = Column(Integer, primary_key=True)
    name = Column(String)
    category = Column(String)  # electronics, beauty, books, fitness
    price = Column(Float)
    cost = Column(Float)
    margin_percent = Column(Float)  # Profit margin percentage

class Order(Base):
    """Customer orders with payment and status tracking"""
    __tablename__ = 'orders'
    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'))
    order_date = Column(Date)
    total_amount = Column(Float)
    status = Column(String)  # completed, cancelled, refunded
    payment_method = Column(String)
    discount_used = Column(Boolean)

class OrderItem(Base):
    """Individual items within each order"""
    __tablename__ = 'order_items'
    order_item_id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'))
    product_id = Column(Integer, ForeignKey('products.product_id'))
    quantity = Column(Integer)
    unit_price = Column(Float)
    discount = Column(Float)

class Interaction(Base):
    """Customer engagement tracking for churn prediction"""
    __tablename__ = 'interactions'
    interaction_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'))
    interaction_date = Column(Date)
    type = Column(String)  # support_ticket, email_open, website_visit
    outcome = Column(String)  # resolved, opened, no_response, clicked

def generate_data():
    """Generate realistic e-commerce data for ML model training"""
    
    # Create 50 products with realistic pricing and margins
    products = []
    for _ in range(50):
        cost = round(random.uniform(5, 50), 2)  # Product cost
        price = round(cost * random.uniform(1.2, 2.0), 2)  # 20-100% markup
        margin = round(((price - cost) / price) * 100, 2)  # Calculate profit margin
        products.append(Product(
            name=fake.word().capitalize(),
            category=random.choice(['electronics', 'beauty', 'books', 'fitness']),
            price=price,
            cost=cost,
            margin_percent=margin
        ))
    session.add_all(products)
    session.commit()

    # Create 500 customers with signup dates over past 2 years
    customers = []
    for _ in range(500):
        signup = fake.date_between(start_date='-2y', end_date='today')
        customers.append(Customer(
            email=fake.email(),
            signup_date=signup,
            tier=random.choice(['bronze', 'silver', 'gold']),  # Customer segments
            country=fake.country(),
            total_orders=0,  # Will be calculated below
            lifetime_value=0.0  # Will be calculated below
        ))
    session.add_all(customers)
    session.commit()

    # Generate orders and order items for each customer
    orders = []
    order_items = []
    for cust in session.query(Customer).all():
        # Each customer gets 1-10 orders
        num_orders = random.randint(1, 10)
        cust.total_orders = num_orders
        
        for _ in range(num_orders):
            # Order date must be after customer signup
            order_date = fake.date_between(start_date=cust.signup_date, end_date='today')
            
            # Most orders complete successfully (80% success rate)
            status = random.choices(['completed', 'cancelled', 'refunded'], weights=[0.8, 0.1, 0.1])[0]
            
            order = Order(
                customer_id=cust.customer_id,
                order_date=order_date,
                total_amount=0.0,  # Will calculate below
                status=status,
                payment_method=random.choice(['card', 'paypal', 'apple_pay']),
                discount_used=random.choice([True, False])
            )
            session.add(order)
            session.flush()  # Get order_id before creating order items

            # Add 1-3 products per order
            total = 0.0
            for _ in range(random.randint(1, 3)):
                product = random.choice(products)
                qty = random.randint(1, 5)
                discount = round(random.uniform(0, 0.3), 2)  # 0-30% discount
                price = product.price * qty * (1 - discount)
                
                order_items.append(OrderItem(
                    order_id=order.order_id,
                    product_id=product.product_id,
                    quantity=qty,
                    unit_price=product.price,
                    discount=discount
                ))
                total += price
            
            # Update order total and customer lifetime value
            order.total_amount = round(total, 2)
            cust.lifetime_value += total

    session.commit()
    session.add_all(order_items)
    session.commit()

    # Generate customer interactions for engagement tracking
    types = ['support_ticket', 'email_open', 'website_visit']
    outcomes = ['resolved', 'opened', 'no_response', 'clicked']
    interactions = []
    
    for cust in session.query(Customer).all():
        # Each customer has 1-5 interactions over past year
        for _ in range(random.randint(1, 5)):
            interactions.append(Interaction(
                customer_id=cust.customer_id,
                interaction_date=fake.date_between(start_date='-1y', end_date='today'),
                type=random.choice(types),
                outcome=random.choice(outcomes)
            ))
    
    session.add_all(interactions)
    session.commit()

if __name__ == "__main__":
    generate_data()
    print("Sample data generated and inserted into revenue_db!")