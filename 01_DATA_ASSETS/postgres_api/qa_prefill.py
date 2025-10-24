import psycopg2
from sentence_transformers import SentenceTransformer
import sys

default_qa = [
    (
        "How to configure ACL with object-groups for multiple services (HTTP HTTPS SSH)?",
        "object-group service ALLOWED_SERVICES tcp \n port-object eq 80 \n port-object eq 443 \n port-object eq 22 \n ip access-list extended FILTER_GROUP \n permit tcp any any object-group ALLOWED_SERVICES \n deny ip any any"
    ),
    (
        "How to configure ACL to block all traffic from 203.0.113.0/24 and log it?",
        "ip access-list extended BLOCK_ATTACKERS \n deny ip 203.0.113.0 0.0.0.255 any log \n permit ip any any"
    ),
    (
        "How to configure a DHCP pool with default gateway 2 DNS servers and domain name and a custom lease of 2 days?",
        "ip dhcp pool ENTERPRISE \n network 10.100.0.0 255.255.255.0 \n default-router 10.100.0.1 \n dns-server 8.8.8.8 1.1.1.1 \n domain-name enterprise.local \n lease 2"
    ),
    (
        "How to configure DHCP snooping on VLANs 10 and 20 trusting uplink GigabitEthernet0/1?",
        "ip dhcp snooping \n ip dhcp snooping vlan 10 20 \n interface GigabitEthernet0/1 \n ip dhcp snooping trust"
    ),
    (
        "How to configure an ACL to allow only SSH (22) and HTTPS (443) from 192.168.10.0/24 to 10.0.0.0/24?",
        "ip access-list extended SECURE_ONLY\n permit tcp 192.168.10.0 0.0.0.255 10.0.0.0 0.0.0.255 eq 22 \n permit tcp 192.168.10.0 0.0.0.255 10.0.0.0 0.0.0.255 eq 443 \n deny ip any any"
    ),
    (
        "How to configure a DHCP pool with two DNS servers (8.8.8.8 and 8.8.4.4) and lease time of 12 hours?",
        "ip dhcp pool USERS \n network 192.168.50.0 255.255.255.0 \n default-router 192.168.50.1 \n dns-server 8.8.8.8 8.8.4.4 \n lease 0 12"
    ),
    (
        "How to configure ACL for denying and logging all denied packets from subnet 172.16.20.0/24?",
        "ip access-list extended DENY_LOG \n deny ip 172.16.20.0 0.0.0.255 any log \n permit ip any any"
    ),
    (
        "How to configure DHCP option 66 (TFTP server name tftp.example.com) and option 150 (server IP 10.10.10.5) for a VOIP pool?",
        "ip dhcp pool VOIP \n network 192.168.60.0 255.255.255.0 \n default-router 192.168.60.1 \n option 66 ascii tftp.example.com \n option 150 ip 10.10.10.5"
    ),
    (
        "How to configure DHCP relay agent with multiple servers (10.1.1.1 and 10.2.2.2) on VLAN 30 interface?",
        "interface Vlan30 \n ip address 192.168.30.1 255.255.255.0 \n ip helper-address 10.1.1.1 \n ip helper-address 10.2.2.2"
    ),
    (
        "How to configure an ACL to block all traffic from subnet 192.168.100.0/24 except DNS (UDP 53)?",
        "ip access-list extended BLOCK_100 \n permit udp 192.168.100.0 0.0.0.255 any eq 53 \n deny ip 192.168.100.0 0.0.0.255 any \n permit ip any any"
    ),
    (
        "How to configure DHCP for multiple VLANs (10 and 20)?",
        "ip dhcp pool VLAN10 \n network 192.168.10.0 255.255.255.0 \n default-router 192.168.10.1 \n ip dhcp pool VLAN20 \n network 192.168.20.0 255.255.255.0 \n default-router 192.168.20.1"
    ),
    (
        "How to configure a DHCP relay agent on interface GigabitEthernet0/1 with server 10.0.0.5?",
        "interface GigabitEthernet0/1 \n ip helper-address 10.0.0.5"
    ),
    (
        "How to configure an ACL that allows only 192.168.10.0/24 to access 10.0.0.1 via SSH?",
        "ip access-list extended SSH_ONLY \n permit tcp 192.168.10.0 0.0.0.255 host 10.0.0.1 eq 22 \n deny ip any host 10.0.0.1 \n permit ip any any"
    ),
    (
        "How to configure OSPF to redistribute connected routes?",
        "router ospf 1 \n redistribute connected subnets"
    ),
    (
        "How to configure OSPF authentication (MD  key 12345) on interface GigabitEthernet0/1?",
        "interface GigabitEthernet0/1 \n ip ospf authentication message-digest \n ip ospf message-digest-key 1 md5 12345"
    ),
    (
        "What is the command to enable OSPF under VRF custA with router ID 10.0.0.103?",
        "router ospf 1 vrf custA \n router-id 10.0.0.103 area 0"
    ),
    (
        "What is the command used to dynamically assign an IP address to interface FastEthernet0/0/0 via DHCP?",
        "interface FastEthernet0/0/0 \n ip address dhcp \n no shutdown"
    ),
    (
        "What is the OSPF configuration for router 10.2.10.10 to advertise networks 172.16.102.0/24",
        "router ospf 1 \n router-id 10.2.10.10 \n network 172.16.102.0 0.0.0.255 area 0"
    )
]

def upsert_qa(question, answer, cur, model):
    embedding = model.encode(question).tolist()
    # UPSERT: update answer and embedding if question exists
    cur.execute("""
        INSERT INTO qa (question, answer, embedding, lastUpdated)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (question) DO UPDATE
        SET answer = EXCLUDED.answer,
            embedding = EXCLUDED.embedding,
            lastUpdated = CURRENT_TIMESTAMP
    """, (question, answer, embedding))

def main():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="/var/run/postgresql",  # Unix socket
        # host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if len(sys.argv) == 3:
        question = sys.argv[1]
        answer = sys.argv[2]
        upsert_qa(question, answer, cur, model)
    else:
        for question, answer in default_qa:
            upsert_qa(question, answer, cur, model)

    conn.commit()
    cur.close()
    conn.close()
    print("QA saved in PostgreSQL.")

if __name__ == "__main__":
    main()
